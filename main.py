import os
import sys
import re
import csv
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Görev Motorları
from gorev1 import DetectionManager
from gorev2 import VOEstimator
try:
    from gorev3 import ReferenceManager
    HAS_GOREV3 = True
except Exception as e:
    ReferenceManager = None
    HAS_GOREV3 = False

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def natural_sort_key(s):
    """Dosya adlarını mantıksal/sayısal sıraya göre sıralar (örn: Referans_Nesne_02, 10'dan önce gelir)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def configure_logger(output_dir="./_logs"):
    """Loglama sistemini yapılandırır. Hem dosyaya hem konsola log yazar."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    log_file = os.path.join(output_dir, f"calisma_{timestamp_str}.log")
    json_file = os.path.join(output_dir, f"sonuclar_{timestamp_str}.json")

    logger = logging.getLogger("AircraftAI")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Dosya Loglayıcısı (UTF-8)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Konsol Loglayıcısı
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger, log_file, json_file


def get_dataset_files(dataset_dir="veriseti"):
    """
    Veriseti klasöründeki görüntüleri veya video dosyasını bulur.
    Doğrudan veriseti/ altında veya alt klasörlerdeki görselleri arar.
    Referans klasörlerini kare listesinden ayıklar.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return [], None

    ref_dir_names = {'references', 'referanslar', 'reference', 'ref'}

    # 1. Doğrudan veriseti/ kök dizinindeki resimler
    direct_images = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if direct_images:
        return sorted(direct_images, key=lambda p: natural_sort_key(p.name)), None

    # 2. Alt klasörleri kontrol et (örn: veriseti/images/ veya veriseti/frames/)
    for sub in dataset_path.iterdir():
        if sub.is_dir() and sub.name.lower() not in ref_dir_names:
            sub_images = [
                f for f in sub.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if sub_images:
                return sorted(sub_images, key=lambda p: natural_sort_key(p.name)), None

    # 3. Tüm alt klasörlerde özyinelemeli arama (referans klasörleri hariç)
    all_images = []
    for f in dataset_path.rglob('*'):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            parts_lower = [p.lower() for p in f.parts]
            if not any(rd in parts_lower for rd in ref_dir_names):
                all_images.append(f)
    if all_images:
        return sorted(all_images, key=lambda p: natural_sort_key(p.name)), None

    # 4. Video dosyası var mı?
    video_files = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if video_files:
        return [], video_files[0]

    return [], None


def get_reference_files(dataset_dir="veriseti"):
    """
    Görev 3 için referans görsellerini bulur ve sıralı sözlük olarak döner.
    veriseti/references/, veriseti/referanslar/ vb. klasörleri arar.
    """
    dataset_path = Path(dataset_dir)
    ref_dir_names = ['references', 'referanslar', 'reference', 'ref']
    ref_images = {}

    for ref_dir_name in ref_dir_names:
        r_dir = dataset_path / ref_dir_name
        if r_dir.exists() and r_dir.is_dir():
            for f in sorted(r_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    img = cv2.imread(str(f))
                    if img is not None:
                        ref_images[f.name] = {
                            "image": img,
                            "path": str(f)
                        }
            if ref_images:
                break

    return ref_images


def get_telemetry_data(dataset_dir="veriseti"):
    """
    Görev 2 için telemetri/yer gerçeği (GT) verilerini (translations.csv/.json) yükler.
    Noktalı virgül (;) ve virgül (,) ayırıcılarını otomatik algılar.
    Kare numarası, dosya adı ve indeks bazında çoklu indeksleme yapar.
    """
    dataset_path = Path(dataset_dir)
    telemetry = {}

    def parse_float_val(v):
        if v is None:
            return None
        v_str = str(v).strip()
        if not v_str:
            return None
        try:
            return float(v_str.replace(',', '.'))
        except ValueError:
            return None

    # CSV Dosyaları
    csv_candidates = ['translations.csv', 'telemetry.csv', 'gt.csv']
    for c_name in csv_candidates:
        c_path = dataset_path / c_name
        if not c_path.exists():
            continue
        try:
            with open(c_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ','
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter)
                for idx, row in enumerate(reader):
                    cleaned_row = {k.strip().lower(): v.strip() for k, v in row.items() if k}

                    health = cleaned_row.get('health_status') or cleaned_row.get('health') or '1'
                    tx_raw = cleaned_row.get('translation_x') or cleaned_row.get('x') or cleaned_row.get('pos_x')
                    ty_raw = cleaned_row.get('translation_y') or cleaned_row.get('y') or cleaned_row.get('pos_y')
                    tz_raw = cleaned_row.get('translation_z') or cleaned_row.get('z') or cleaned_row.get('pos_z')

                    item = {
                        'health_status': str(health).strip(),
                        'gt_x': parse_float_val(tx_raw),
                        'gt_y': parse_float_val(ty_raw),
                        'gt_z': parse_float_val(tz_raw),
                    }

                    # Çeşitli anahtarlarla erişim imkanı sağla:
                    frame_num_raw = cleaned_row.get('frame_numbers') or cleaned_row.get('frame') or cleaned_row.get('image_name')
                    if frame_num_raw:
                        base_fn = os.path.basename(str(frame_num_raw).strip())
                        telemetry[base_fn] = item
                        telemetry[base_fn.lower()] = item
                        telemetry[f"{base_fn}.jpg"] = item
                        telemetry[f"{base_fn}.png"] = item

                        digits = re.findall(r'\d+', base_fn)
                        if digits:
                            telemetry[str(int(digits[-1]))] = item

                    # Sayısal satır indeksiyle de kaydet
                    telemetry[str(idx)] = item

            if telemetry:
                return telemetry
        except Exception as e:
            logging.warning(f"{c_name} okunurken hata: {e}")

    # JSON Dosyaları
    for json_name in ['translations.json', 'telemetry.json', 'poses.json', 'gt.json']:
        j_path = dataset_path / json_name
        if j_path.exists():
            try:
                with open(j_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        key = item.get('image_url') or item.get('frame') or item.get('image_name') or str(idx)
                        key = os.path.basename(str(key))
                        entry = {
                            'health_status': str(item.get('health_status', '1')),
                            'gt_x': parse_float_val(item.get('translation_x', item.get('x'))),
                            'gt_y': parse_float_val(item.get('translation_y', item.get('y'))),
                            'gt_z': parse_float_val(item.get('translation_z', item.get('z'))),
                        }
                        telemetry[key] = entry
                        telemetry[str(idx)] = entry
                elif isinstance(data, dict):
                    for k, v in data.items():
                        base_k = os.path.basename(str(k))
                        entry = {
                            'health_status': str(v.get('health_status', '1')),
                            'gt_x': parse_float_val(v.get('translation_x', v.get('x'))),
                            'gt_y': parse_float_val(v.get('translation_y', v.get('y'))),
                            'gt_z': parse_float_val(v.get('translation_z', v.get('z'))),
                        }
                        telemetry[base_k] = entry
                return telemetry
            except Exception as e:
                logging.warning(f"{json_name} okunurken hata: {e}")

    return telemetry


def prompt_reference_intervals(ref_images, total_frames):
    """
    Kullanıcıdan her bir referans nesne için aranacak kare aralığını terminalde sorar.
    Sırayla sorulur.
    Formatlar: '100 500', '100-500' veya doğrudan Enter (atla).
    Dönüş: {ref_name: (start_frame, end_frame)}
    """
    if not ref_images:
        return {}

    ref_names = sorted(list(ref_images.keys()), key=natural_sort_key)
    intervals = {}
    max_frame_idx = max(0, total_frames - 1)

    print("\n" + "=" * 68)
    print("           GÖREV 3: REFERANS NESNE ARAMA ARALIKLARI BELİRLEME        ")
    print("=" * 68)
    print(f"Toplam {len(ref_names)} adet referans nesnesi bulundu.")
    print(f"Veriseti Kare İndeksleri: 0 - {max_frame_idx} (Toplam {total_frames} kare)")
    print("\nHer referans nesnenin aranacağı kare aralığını girin.")
    print("Örnek girişler:")
    print("  • '100 500' veya '100-500' -> 100 ile 500. kareler arasında arar")
    print("  • 'hepsi' veya 'all'       -> Tüm videoda (0 - {max_frame_idx}) arar")
    print("  • [Boş / Enter tuşu]       -> Bu referansı atlar (hiç aramaz)")
    print("-" * 68)

    for i, ref_name in enumerate(ref_names):
        # Etkileşimli terminal kontrolü
        if not sys.stdin.isatty():
            # Otomatik test veya arka plan çalışması durumunda tüm aralığı varsay
            intervals[ref_name] = (0, max_frame_idx)
            continue

        while True:
            try:
                user_input = input(f"[{i+1}/{len(ref_names)}] '{ref_name}' aralığı: ").strip()

                if not user_input:
                    print(f"   -> ⏭️ '{ref_name}' ATLANDI (aranmayacak).")
                    intervals[ref_name] = None
                    break

                if user_input.lower() in ['tümü', 'hepsi', 'all', '*', 'tum']:
                    intervals[ref_name] = (0, max_frame_idx)
                    print(f"   -> ✅ '{ref_name}': Tüm karelerde aranacak (0 - {max_frame_idx}).")
                    break

                nums = re.findall(r'\d+', user_input)
                if len(nums) >= 2:
                    s, e = int(nums[0]), int(nums[1])
                    start_f, end_f = min(s, e), max(s, e)
                    intervals[ref_name] = (start_f, end_f)
                    print(f"   -> ✅ '{ref_name}': Frame {start_f} - {end_f} arasında aranacak.")
                    break
                elif len(nums) == 1:
                    start_f = int(nums[0])
                    end_raw = input(f"   -> Bitiş karesi (başlangıç {start_f}): ").strip()
                    end_nums = re.findall(r'\d+', end_raw)
                    if end_nums:
                        end_f = int(end_nums[0])
                        start_f, end_f = min(start_f, end_f), max(start_f, end_f)
                        intervals[ref_name] = (start_f, end_f)
                        print(f"   -> ✅ '{ref_name}': Frame {start_f} - {end_f} arasında aranacak.")
                        break
                    else:
                        print("   ⚠️ Geçersiz bitiş karesi. Lütfen tekrar deneyin.")
                else:
                    print("   ⚠️ Lütfen iki sayı girin (örn: 100 500) veya atlamak için Enter'a basın.")
            except (EOFError, KeyboardInterrupt):
                print(f"\n   -> Giriş sonlandırıldı. '{ref_name}' atlandı.")
                intervals[ref_name] = None
                break

    print("-" * 68)
    active_refs = {k: v for k, v in intervals.items() if v is not None}
    print(f"Arama Yapılacak Aktif Referans Sayısı: {len(active_refs)}")
    for k, v in active_refs.items():
        print(f"  • {k:<25} : Frame {v[0]} - {v[1]}")
    print("=" * 68 + "\n")

    return intervals


def run(dataset_dir="veriseti"):
    logger, log_file, json_file = configure_logger()
    logger.info("=================================================================")
    logger.info("               HAVACILIKTA YAPAY ZEKA PİPELİNE'I BAŞLATILDI      ")
    logger.info("=================================================================")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        dataset_path.mkdir(parents=True, exist_ok=True)
        msg = (
            f"⚠️ '{dataset_dir}' klasörü bulunamadı! Klasör otomatik olarak oluşturuldu:\n"
            f"   -> {dataset_path.resolve()}\n\n"
            "Lütfen test etmek istediğiniz görsel/video ve csv verilerini bu klasöre yerleştirin.\n"
        )
        logger.warning(msg)
        return

    # Verisetini tara
    image_files, video_file = get_dataset_files(dataset_dir)

    if not image_files and not video_file:
        msg = (
            f"⚠️ '{dataset_dir}' klasöründe işlenecek görsel (.jpg, .png vb.) veya video (.mp4 vb.) bulunamadı!\n"
            f"Lütfen verilerinizi '{dataset_dir}' klasörüne ekleyip tekrar çalıştırın."
        )
        logger.warning(msg)
        return

    # Toplam kare sayısını belirle
    total_frames = 0
    if image_files:
        total_frames = len(image_files)
    elif video_file:
        cap_temp = cv2.VideoCapture(str(video_file))
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap_temp.release()

    # Görev 3 Referans görsellerini bul
    ref_images = get_reference_files(dataset_dir)

    # Kullanıcıdan her referans için aralıkları terminalde sor
    ref_intervals = prompt_reference_intervals(ref_images, total_frames)
    active_ref_names = [k for k, v in ref_intervals.items() if v is not None]

    if active_ref_names:
        logger.info(f"🎯 Görev 3: {len(active_ref_names)} aktif referans nesnesi aranacak: {active_ref_names}")
    else:
        logger.info("ℹ️ Görev 3: Aktif referans nesnesi seçilmedi veya referans bulunamadı. Görev 3 atlanacak.")

    # Görev 2 Telemetri / CSV verilerini oku
    telemetry_data = get_telemetry_data(dataset_dir)
    if telemetry_data:
        logger.info(f"📡 Görev 2: Telemetri/CSV verisi yüklendi ({len(telemetry_data)} kayıt eşleştirildi).")
    else:
        logger.warning("⚠️ Görev 2: 'translations.csv' veya telemetri verisi bulunamadı!")

    # Motorları Başlat
    logger.info("🚀 AI Görev Motorları yükleniyor...")
    det_manager = DetectionManager()
    vo_estimator = VOEstimator()

    ref_manager = None
    if active_ref_names and HAS_GOREV3:
        try:
            logger.info("🚀 Görev 3 Referans Motoru (DINOv2) başlatılıyor...")
            ref_manager = ReferenceManager()
        except Exception as e:
            logger.error(f"Görev 3 Motoru başlatılamadı: {e}")
            ref_manager = None

    all_frame_results = []
    class_counts = {}
    total_detections = 0
    total_ref_matches = 0

    pipeline_start_time = time.perf_counter()

    # Geçici kare kaydetme klasörü (video okuma için VOEstimator disk yolu bekler)
    temp_dir = Path("./_logs/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_frame_path = temp_dir / "current_frame.jpg"

    def process_single_frame(idx, frame, frame_name, frame_file_path):
        nonlocal total_detections, total_ref_matches
        t_start = time.perf_counter()

        # -------------------------------------------------------------
        # 1. GÖREV 1: NESNE TESPİTİ (YOLO & Optik Akış)
        # -------------------------------------------------------------
        detections = det_manager.process_frame(frame)
        for item in detections:
            cls_k = item.get("cls_key", "Bilinmeyen")
            class_counts[cls_k] = class_counts.get(cls_k, 0) + 1
            total_detections += 1

        # -------------------------------------------------------------
        # 2. GÖREV 2: KONUM KESTİRİMİ (VO & CSV DESTEĞİ)
        # -------------------------------------------------------------
        # Telemetri arama: dosya adı, uzantısız ad veya sayısal indeks
        telem = (telemetry_data.get(frame_name) or
                 telemetry_data.get(os.path.splitext(frame_name)[0]) or
                 telemetry_data.get(str(idx)))

        if telem:
            gt_x = telem.get('gt_x')
            gt_y = telem.get('gt_y')
            gt_z = telem.get('gt_z')
            health_status = str(telem.get('health_status', '1')).strip()
        else:
            gt_x = gt_y = gt_z = None
            health_status = None

        # Kural:
        # health_status == '1' -> CSV dosyasındaki tablodaki değerler verilir (ve VO bu GT ile eğitilir/güncellenir)
        # health_status == '0' -> gorev2.py daki VO motorunun hesapladığı tahmin değerleri verilir
        if health_status == '1':
            vo_estimator.process_frame(str(frame_file_path), gt_x, gt_y, gt_z, '1')
            out_x = gt_x
            out_y = gt_y
            out_z = gt_z
            pos_source = "CSV Tablosu (GT)"
        elif health_status == '0':
            est_x, est_y, est_z = vo_estimator.process_frame(str(frame_file_path), None, None, None, '0')
            out_x = float(est_x)
            out_y = float(est_y)
            out_z = float(est_z)
            pos_source = "Görev 2 VO Tahmini"
        else:
            est_x, est_y, est_z = vo_estimator.process_frame(str(frame_file_path), None, None, None, None)
            out_x = float(est_x)
            out_y = float(est_y)
            out_z = float(est_z)
            pos_source = "VO (Telemetri Yok)"

        # -------------------------------------------------------------
        # 3. GÖREV 3: REFERANS NESNE ARAMA (Sırayla ve Aralık Kontrolüyle)
        # -------------------------------------------------------------
        ref_results = []
        if ref_manager is not None:
            # Sırayla tüm aktif referansları kontrol et
            for ref_name in active_ref_names:
                interval = ref_intervals.get(ref_name)
                if not interval:
                    continue

                start_frame, end_frame = interval
                # Yalnızca bu referansın belirtilen aralığında ara
                if start_frame <= idx <= end_frame:
                    ref_img = ref_images[ref_name]["image"]
                    bbox = ref_manager.find_reference(frame, ref_img, ref_name, is_thermal=False)
                    if bbox:
                        ref_results.append({
                            "reference_name": ref_name,
                            "bbox": [int(b) for b in bbox]
                        })
                        total_ref_matches += 1

        t_elapsed = time.perf_counter() - t_start

        # -------------------------------------------------------------
        # LOGLAMA
        # -------------------------------------------------------------
        logger.info(f"[Kare {idx + 1}/{total_frames}] '{frame_name}' işlendi ({t_elapsed:.3f}s)")

        # Görev 1 Logu
        if detections:
            logger.info(f"  └── Görev 1 (Nesne Tespiti - {len(detections)} adet):")
            for d in detections:
                logger.info(f"      • [{d['cls_key']}] BBox: {d['bbox']} | İniş: {d['landing_key']} | Hareket: {d['moving_key']}")
        else:
            logger.info("  └── Görev 1: Nesne tespit edilmedi.")

        # Görev 2 Logu
        if out_x is not None:
            if health_status == '1':
                logger.info(f"  └── Görev 2 (Konum): X={out_x:.4f}, Y={out_y:.4f}, Z={out_z:.4f} [Sağlık: 1 - {pos_source}]")
            elif health_status == '0':
                gt_info = f" (CSV GT: {gt_x:.4f}, {gt_y:.4f}, {gt_z:.4f})" if gt_x is not None else ""
                logger.info(f"  └── Görev 2 (Konum): X={out_x:.4f}, Y={out_y:.4f}, Z={out_z:.4f} [Sağlık: 0 - {pos_source}{gt_info}]")
            else:
                logger.info(f"  └── Görev 2 (Konum): X={out_x:.4f}, Y={out_y:.4f}, Z={out_z:.4f} [{pos_source}]")

        # Görev 3 Logu
        if active_ref_names:
            if ref_results:
                logger.info(f"  └── Görev 3 (Referans - {len(ref_results)} eşleşme):")
                for r in ref_results:
                    logger.info(f"      • Referans: '{r['reference_name']}' -> BBox: {r['bbox']}")
            else:
                # O karede aranan aktif referanslar var mıydı?
                currently_searched = [rn for rn in active_ref_names if ref_intervals[rn][0] <= idx <= ref_intervals[rn][1]]
                if currently_searched:
                    logger.info(f"  └── Görev 3: {currently_searched} arandı, eşleşme bulunamadı.")

        frame_record = {
            "frame_index": idx,
            "frame_name": frame_name,
            "elapsed_seconds": round(t_elapsed, 4),
            "detections": detections,
            "translation": {
                "x": round(out_x, 4) if out_x is not None else None,
                "y": round(out_y, 4) if out_y is not None else None,
                "z": round(out_z, 4) if out_z is not None else None,
                "health_status": health_status,
                "source": pos_source
            },
            "references": ref_results
        }
        all_frame_results.append(frame_record)

    # -------------------------------------------------------------
    # ANA İŞLEME DÖNGÜSÜ
    # -------------------------------------------------------------
    try:
        if image_files:
            logger.info(f"📊 Toplam {total_frames} adet görsel dosya işlenecek.")
            with tqdm(total=total_frames, desc="Görseller İşleniyor") as pbar:
                for idx, img_path in enumerate(image_files):
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        logger.warning(f"⚠️ Görüntü okunamadı: {img_path.name}")
                        pbar.update(1)
                        continue

                    process_single_frame(idx, frame, img_path.name, img_path)
                    pbar.update(1)

        elif video_file:
            logger.info(f"🎬 Video dosyası işleniyor: {video_file.name} ({total_frames} kare)")
            cap = cv2.VideoCapture(str(video_file))
            idx = 0

            with tqdm(total=total_frames if total_frames > 0 else None, desc="Video İşleniyor") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_name = f"frame_{idx:06d}.jpg"
                    # VO motorunun dosya yolundan okuyabilmesi için kareyi geçici dosyaya yaz
                    cv2.imwrite(str(temp_frame_path), frame)

                    process_single_frame(idx, frame, frame_name, temp_frame_path)

                    idx += 1
                    pbar.update(1)

                cap.release()
                total_frames = idx

    except KeyboardInterrupt:
        logger.warning("\n⚠️ İşlem kullanıcı tarafından durduruldu (Ctrl+C). İşlenen kareler kaydediliyor...")

    finally:
        # Geçici dosyayı temizle
        if temp_frame_path.exists():
            try:
                os.remove(temp_frame_path)
                temp_dir.rmdir()
            except Exception:
                pass

    total_pipeline_elapsed = time.perf_counter() - pipeline_start_time
    processed_count = len(all_frame_results)
    avg_fps = (processed_count / total_pipeline_elapsed) if total_pipeline_elapsed > 0 else 0.0

    # JSON Olarak Kaydet
    output_summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_directory": str(dataset_dir),
        "total_frames_processed": processed_count,
        "total_elapsed_seconds": round(total_pipeline_elapsed, 2),
        "average_fps": round(avg_fps, 2),
        "reference_intervals": {k: list(v) if v else None for k, v in ref_intervals.items()},
        "summary": {
            "total_detections": total_detections,
            "class_counts": class_counts,
            "total_reference_matches": total_ref_matches
        },
        "frames": all_frame_results
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_summary, f, ensure_ascii=False, indent=2)

    # Özet Raporunu Logla ve Ekrana Bas
    logger.info("=================================================================")
    logger.info("                  İŞLEM ÖZETİ VE İSTATİSTİKLER                   ")
    logger.info("=================================================================")
    logger.info(f"Toplam İşlenen Kare Sayısı : {processed_count}")
    logger.info(f"Toplam Geçen Süre          : {total_pipeline_elapsed:.2f} saniye")
    logger.info(f"Ortalama İşlem Hızı        : {avg_fps:.2f} FPS")
    logger.info("-----------------------------------------------------------------")
    logger.info("Görev 1 (Nesne Tespiti) Sınıf İstatistikleri:")
    if class_counts:
        for cls_name, count in class_counts.items():
            logger.info(f"  • {cls_name:<12}: {count} adet")
        logger.info(f"  • Toplam Tespit: {total_detections} adet")
    else:
        logger.info("  • Hiç nesne tespit edilmedi.")
    logger.info("-----------------------------------------------------------------")
    if active_ref_names:
        logger.info(f"Görev 3 (Referans Tespiti) Eşleşme Sayısı: {total_ref_matches} adet")
        logger.info("-----------------------------------------------------------------")
    logger.info(f"Log Dosyası    : {os.path.abspath(log_file)}")
    logger.info(f"JSON Sonuçları : {os.path.abspath(json_file)}")
    logger.info("=================================================================")


if __name__ == '__main__':
    run()
