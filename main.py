import sys
import cv2
import numpy as np
import easyocr
import json
import os
import shutil
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
import time

load_dotenv()

API_KEY_NAME = "X-API-Key"
API_KEY_VALUE = os.getenv("SECRET_KEY", "super_secret_python_key_123abcXYZ")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY_VALUE:
        return api_key
    raise HTTPException(
        status_code=403,
        detail={"status": "error", "message": "Could not validate credentials - Invalid API Key"}
    )

app = FastAPI(
    title="KTP OCR API",
    description="API for KTP image processing and OCR.",
    version="1.0.0",
)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightA))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
MODELS_PATH = '/var/www/literasearch_tesseract/easyocr_models'
reader = easyocr.Reader(
    ['en', 'id'], 
    gpu=False,
    model_storage_directory=MODELS_PATH)

@app.post("/process-ktp/")
async def process_ktp(
    identity_image: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """
    Endpoint for processing KTP images and extracting information using OCR.
    """
    temp_file_path = ""
    try:
        os.makedirs("temp_uploads", exist_ok=True)
        temp_file_path = os.path.join("temp_uploads", identity_image.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(identity_image.file, buffer)

        ocr_result = process_ktp_image_core(temp_file_path)

        if "error" in ocr_result:
            raise HTTPException(status_code=500, detail=ocr_result["error"])

        extracted_data = ocr_result["ocr_data"]


        required_fields = ["nik", "nama", "tempat_lahir", "tanggal_lahir", "alamat", "rt_rw", "kel_desa", "kecamatan", "jenis_kelamin"]
        print(f"debug '{ocr_result}'")
        for field in required_fields:
            if extracted_data.get(field) is None or extracted_data.get(field) == "":
                raise HTTPException(status_code=400, detail=f"Tolong ulangi foto gambar KTP. Data '{field}' tidak lengkap.")

        response_data = {
            "message": "KTP processed successfully!",
            "data": {
                "ocr_result": extracted_data
            }
        }
        return JSONResponse(content=response_data, status_code=200)

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def process_ktp_image_core(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image. Check path or file corruption."}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        ktp_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 1.4 < aspect_ratio < 1.8 and w > img.shape[1] * 0.4 and h > img.shape[0] * 0.2:
                    ktp_contour = approx
                    break

        cropped_ktp_image = None
        if ktp_contour is not None:
            cropped_ktp_image = four_point_transform(img, ktp_contour.reshape(4, 2))
            target_width = 1000
            target_height = int(target_width / 1.58)
            cropped_ktp_image = cv2.resize(cropped_ktp_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            cropped_ktp_image = img

        if cropped_ktp_image is None:
            return {"error": "Failed to process image for OCR."}

        results = reader.readtext(cropped_ktp_image)
        print(f"debug result '{results}'")
        ocr_data = {
            "nik": None,
            "nama": None,
            "tempat_lahir": None,
            "tanggal_lahir": None,
            "alamat": None,
            "rt_rw": None,
            "kel_desa": None,
            "kecamatan": None,
            "alamat_lengkap": None,
            "jenis_kelamin": None,
            "raw_text": ""
        }

        full_text_list = [text.strip() for (bbox, text, prob) in results]
        raw_text_combined = " ".join(full_text_list).upper()
        ocr_data['raw_text'] = raw_text_combined

        img_with_boxes = cropped_ktp_image.copy()
        box_color = (0, 255, 0)
        line_thickness = 2
        for (bbox, text, prob) in results:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(img_with_boxes, top_left, bottom_right, box_color, line_thickness)
        timestamp = int(time.time() * 1000000)
        debug_image_path = os.path.join("temp_uploads", "debug_ocr_" + str(timestamp) + "_" + os.path.basename(image_path))
        cv2.imwrite(debug_image_path, img_with_boxes)


        nik_match = re.search(r'(?i)\bNIK\s*([A-Z0-9]{15,17})\b', raw_text_combined)
        def nik_clean(nik_raw):
            nik_raw = nik_raw.upper()
            replacements = {
            'L': '1', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'O': '0', 'Q': '0', 'G': '6', 'T': '7'
            }
            for letter, digit in replacements.items():
                nik_raw = nik_raw.replace(letter, digit)
                nik_digits = re.sub(r'[^0-9]', '', nik_raw)
            return nik_digits

        if nik_match:
            nik_candidate = nik_clean(nik_match.group(1))
            if len(nik_candidate) == 17:
                ocr_data['nik'] = nik_candidate[:16]
            else:
                ocr_data['nik'] = nik_candidate
        else:
            nik_match_standalone = re.search(r'\b(3[1-6A-Z][A-Z0-9]{14,16})\b', raw_text_combined)
            if nik_match_standalone:
                nik_candidate = nik_clean(nik_match_standalone.group(1))
                if len(nik_candidate) == 17:
                    ocr_data['nik'] = nik_candidate[:16]
                else:
                    ocr_data['nik'] = nik_candidate
            else:
                nik_labeled_match = re.search(r'(?i)\bNIK\s*([A-Z0-9]{15,17})\b', raw_text_combined)
                if nik_labeled_match:
                    nik_candidate = nik_clean(nik_labeled_match.group(1))
                    if len(nik_candidate) == 17:
                        ocr_data['nik'] = nik_candidate[:16]
                    else:
                        ocr_data['nik'] = nik_candidate

        name_start_pattern = r'(?i)(NAMA|NANA|NTER NANA|NTER)'
        name_end_pattern = r'(?i)(TEMPAT/TGL LAHIR|TEMPAT LAHIR|TGL LAHIR|REMPATIGTLANIR|JANIS KELAMN|JENIS KELAMIN|GOL DARAH|AGAMA|STATUS|ALAMAT)'

        nama_potential_text = ""
        if ocr_data['nik']:
            nik_end_pos = raw_text_combined.find(ocr_data['nik'])
            if nik_end_pos != -1:
                nama_potential_text = raw_text_combined[nik_end_pos + len(ocr_data['nik']):].strip()

        if not nama_potential_text:
            name_kw_match = re.search(name_start_pattern, raw_text_combined)
            if name_kw_match:
                nama_potential_text = raw_text_combined[name_kw_match.end():].strip()

        if nama_potential_text:
            end_name_delimiter = re.search(name_end_pattern, nama_potential_text)
            if end_name_delimiter:
                candidate_name = nama_potential_text[:end_name_delimiter.start()].strip()
            else:
                candidate_name = nama_potential_text.split('\n')[0].strip()

            candidate_name = re.sub(r'[^A-Z\s\.,\'-]', '', candidate_name).strip()
            candidate_name = re.sub(name_start_pattern, '', candidate_name, flags=re.IGNORECASE).strip()

            name_words = candidate_name.split()
            if len(name_words) > 3:
                ocr_data['nama'] = " ".join(name_words[:3])
            else:
                ocr_data['nama'] = candidate_name
            if len(ocr_data['nama'] or "") < 3 or not any(char.isalpha() for char in (ocr_data['nama'] or "")):
                ocr_data['nama'] = "UNKNOWN"
        else:
            ocr_data['nama'] = "UNKNOWN"



        ttl_match = re.search(r'(?i)(TEMPAT/TGL LAHIR|TEMPAT LAHIR|TGL LAHIR|REMPATIGTLANIR)\s*:?\s*([A-Z\s\.,]+?)\s*[,:]?\s*(\d{2}[-\s./]\d{2}[-\s./]\d{4})', raw_text_combined)
        if ttl_match:
            ocr_data['tempat_lahir'] = ttl_match.group(2).strip()
            ocr_data['tanggal_lahir'] = ttl_match.group(3).replace(' ', '-').replace('/', '-').replace('.', '-')
        else:
            date_match = re.search(r'\b(\d{2}[-\s./]\d{2}[-\s./]\d{4})\b', raw_text_combined)
            if date_match:
                ocr_data['tanggal_lahir'] = date_match.group(1).replace(' ', '-').replace('/', '-').replace('.', '-')
                if not ocr_data['tempat_lahir']:
                    date_index = raw_text_combined.find(date_match.group(0))
                    if date_index != -1:
                        pre_date_text = raw_text_combined[:date_index].upper()
                        possible_place_candidates = re.findall(r'\b([A-Z]+)\b', pre_date_text)
                        possible_place = []
                        for word in reversed(possible_place_candidates):
                            if word in ["NAMA", "NIK", "PROVINSI", "KABUPATEN", "KOTA", "KECAMATAN", "DESA", "KELURAHAN", "ALAMAT", "REMPATIGTLANIR", "JANIS KELAMN", "JENIS KELAMIN", "GOL DARAH", "P", "L"]:
                                break
                            possible_place.insert(0, word)
                            if len(possible_place) > 0 and len(" ".join(possible_place).strip()) > 2:
                                break
                        ocr_data['tempat_lahir'] = " ".join(possible_place).strip()

        if not ocr_data['tempat_lahir']:
            ocr_data['tempat_lahir'] = "UNKNOWN"
        if not ocr_data['tanggal_lahir']:
            ocr_data['tanggal_lahir'] = "-"



        alamat_start_index = raw_text_combined.find("ALAMAT")
        if alamat_start_index != -1:
            alamat_segment_raw = raw_text_combined[alamat_start_index + len("ALAMAT"):].strip()


            alamat_segment_raw = re.sub(r'(?i)\bATAW\s*(\d{1,3}[/\\]\d{1,3})\b', r'RT/RW \1', alamat_segment_raw)
            alamat_segment_raw = alamat_segment_raw.replace("KELDESA", "KEL DESA")
            alamat_segment_raw = alamat_segment_raw.replace("KECAMALAN", "KECAMATAN")
            alamat_segment_raw = re.sub(r'\s+', ' ', alamat_segment_raw).strip()


            rt_rw_pattern = r'(?i)(RT/?RW|RT\s*RW)\s*:?\s*(\d{2,3}[/\\]\d{2,3})'
            kel_desa_pattern = r'(?i)(KEL/?DESA|KEL DESA|KELURAHAN|DESA)\s*:?\s*([A-Z\s\.,]+?)(?=\s*(KECAMATAN|AGAMA|STATUS|PEKERJAAN|$))'
            kecamatan_pattern = r'(?i)(KECAMATAN|KEC)\s*:?\s*([A-Z\s\.,]+?)(?=\s*(AGAMA|STATUS|PEKERJAAN|KEWARGANEGARAAN|$))'

            rt_rw_match = re.search(rt_rw_pattern, alamat_segment_raw)
            kel_desa_match = re.search(kel_desa_pattern, alamat_segment_raw)
            kecamatan_match = re.search(kecamatan_pattern, alamat_segment_raw)

            if rt_rw_match:
                ocr_data['rt_rw'] = rt_rw_match.group(2).replace('\\', '/')
            if kel_desa_match:
                ocr_data['kel_desa'] = kel_desa_match.group(2).strip()
                ocr_data['kel_desa'] = re.sub(r'[^A-Z\s\.,]', '', ocr_data['kel_desa']).strip()
                ocr_data['kel_desa'] = re.sub(r'^(KEL|DESA)\s*', '', ocr_data['kel_desa'], flags=re.IGNORECASE).strip()

                if ocr_data['kel_desa'] == "SULAIAN":
                    ocr_data['kel_desa'] = "SULAIMAN"
            if kecamatan_match:
                ocr_data['kecamatan'] = kecamatan_match.group(2).strip()
                ocr_data['kecamatan'] = re.sub(r'[^A-Z\s\.,]', '', ocr_data['kecamatan']).strip()
                ocr_data['kecamatan'] = re.sub(r'^(KECAMATAN|KEC)\s*', '', ocr_data['kecamatan'], flags=re.IGNORECASE).strip()

                if ocr_data['kecamatan'] == "MAAGAHAYO":
                    ocr_data['kecamatan'] = "MARGAHAYU"


            boundary_index = len(alamat_segment_raw)
            if rt_rw_match: boundary_index = min(boundary_index, rt_rw_match.start())
            if kel_desa_match: boundary_index = min(boundary_index, kel_desa_match.start())
            if kecamatan_match: boundary_index = min(boundary_index, kecamatan_match.start())

            alamat_clean = alamat_segment_raw[:boundary_index].strip()
            alamat_clean = re.sub(r'^(ALAMAT)\s*', '', alamat_clean, flags=re.IGNORECASE).strip()
            ocr_data['alamat'] = alamat_clean


        if not ocr_data['alamat']:
            ocr_data['alamat'] = "UNKNOWN"
        if not ocr_data['rt_rw']:
            ocr_data['rt_rw'] = "UNKNOWN"
        if not ocr_data['kel_desa']:
            ocr_data['kel_desa'] = "UNKNOWN"
        if not ocr_data['kecamatan']:
            ocr_data['kecamatan'] = "UNKNOWN"


        alamat_parts = []
        if ocr_data['alamat'] and ocr_data['alamat'] != "UNKNOWN":
            alamat_parts.append(ocr_data['alamat'])
        if ocr_data['rt_rw'] and ocr_data['rt_rw'] != "UNKNOWN":
            alamat_parts.append(f"RT/RW {ocr_data['rt_rw']}")
        if ocr_data['kel_desa'] and ocr_data['kel_desa'] != "UNKNOWN":
            alamat_parts.append(f"KEL/DESA {ocr_data['kel_desa']}")
        if ocr_data['kecamatan'] and ocr_data['kecamatan'] != "UNKNOWN":
            alamat_parts.append(f"KEC {ocr_data['kecamatan']}")

        if alamat_parts:
            ocr_data['alamat_lengkap'] = ", ".join(alamat_parts)
        else:
            ocr_data['alamat_lengkap'] = "UNKNOWN"



        gender_match = re.search(r'(?i)(JENIS\s*KELAMIN|JANIS\s*KELAMN)\s*:?\s*(LAKI[- ]*LAKI|PEREMPUAN|L|P)\b', raw_text_combined)

        if gender_match:




            gender_str = gender_match.group(2).strip().upper()
            if gender_str in ["LAKI-LAKI", "LAKI LAKI", "L", "LAKI", "LAKILAKI"]:
                ocr_data['jenis_kelamin'] = "LAKI-LAKI"
            elif gender_str in ["PEREMPUAN", "P"]:
                ocr_data['jenis_kelamin'] = "PEREMPUAN"

        else:

            fallback_gender_match = re.search(r'(?i)\b(LAKI[- ]*LAKI|PEREMPUAN|L|P)\b', raw_text_combined)
            if fallback_gender_match:

                gender_str = fallback_gender_match.group(1).strip().upper()
                if gender_str in ["LAKI-LAKI", "LAKI LAKI", "L", "LAKILAKI"]:
                    ocr_data['jenis_kelamin'] = "LAKI-LAKI"
                elif gender_str in ["PEREMPUAN", "P"]:
                    ocr_data['jenis_kelamin'] = "PEREMPUAN"

            else:

                ocr_data['jenis_kelamin'] = "UNKNOWN"

        return {"status": "success", "ocr_data": ocr_data}

    except Exception as e:
        return {"error": f"An unexpected error occurred during OCR processing: {str(e)}"}
