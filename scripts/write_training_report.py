"""Publish measured run artifacts and a Vietnamese report without inventing scores."""
import argparse
import csv
import json
import shutil
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pct(value):
    return f'{value * 100:.2f}%'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    detector=ROOT/'runs/waste_pilot/yolo26n_multisource30_v1'
    battery=ROOT/'runs/battery_screen/yolo26n_cls_v1'
    summary=read_json(detector/'pilot_summary.json')
    comparison=read_json(detector/'comparison.json')
    b=read_json(battery/'battery_report.json')
    battery_v2=ROOT/'runs/battery_screen/yolo26n_cls_multisource_v2'
    b2=read_json(battery_v2/'battery_report.json')
    external=read_json(battery/'external_recybat_report.json')
    unit_log=(ROOT/'logs/test_standalone_final.log').read_text(encoding='utf-8')
    quality_log=(ROOT/'logs/test_quality_safety.log').read_text(encoding='utf-8')
    passed=sum(int(re.search(r'(\d+) passed', text).group(1)) for text in [unit_log,quality_log])
    epoch_count=len(list(csv.DictReader((detector/'results.csv').open())))
    rows=[]
    for domain,label in [('old_validation','TACO validation cũ (15 ảnh)'),('hf_taco','TACO mirror HF (100 ảnh)'),('kaggle_warp','WaRP validation (200 ảnh)'),('new_validation','Validation hỗn hợp (318 ảnh)'),('reserved_test','Test TACO giữ riêng (15 ảnh)')]:
        old=comparison['pilot'][domain];new=comparison['expanded'][domain]
        key='metrics/mAP50(B)'
        delta=100*(new[key]-old[key])
        rows.append(f'| {label} | {pct(old[key])} | {pct(new[key])} | {delta:+.2f} điểm % | {pct(new["metrics/mAP50-95(B)"])} | {pct(new["metrics/recall(B)"])} |')
    per_class='\n'.join(f'| {row["class_name"]} | {pct(row["precision"])} | {pct(row["recall"])} | {pct(row["mAP50"])} |' for row in summary['per_class'])
    negatives=comparison['expanded']['negative_diagnostic']
    validation=b['validation'];test=b['test']
    scorecard={
        'date':'2026-11-10','scope':'local experimental training and software checks',
        'detector':{'epochs_completed':epoch_count,'metrics':comparison,'production_validated':False},
        'battery_baseline_v1':b,'battery_v1_external':external,'battery_multisource_v2':b2,
        'safety_perception_medical':'not_trained_or_evaluated_in_this_run',
        'radiation':'not_measured; user-reported precautionary guidance only',
        'collection_acceptance_current':'not_confirmed',
        'legacy_classifier_checkpoint_loaded':False,
        'release_gate':'NOT_READY_FOR_UNSUPERVISED_USE',
        'test_logs':{name:(ROOT/'logs'/name).read_text(encoding='utf-8',errors='replace') for name in ['test_standalone_final.log','test_quality_safety.log','test_full_backend_api_running.log','frontend_build_safety.log','frontend_lint_safety.log','pip_check_safety.log']},
    }
    report=f'''# Báo cáo train và mở rộng Đà Nẵng — 10/11/2026

Đã hoàn tất train và đo trên máy RTX 4050. **Các model mới còn ở mức thử nghiệm;
chưa đủ căn cứ dùng tự động để quyết định xử lý chất thải nguy hại.** Không thay
checkpoint MobileNet cũ. Không tạo ảnh AI, không commit/push.

Phát hiện chính: pin v1 accuracy nội bộ 95,59% nhưng recall ngoài miền RecyBat
0%. Vì vậy đã train v2 đa nguồn; holdout phát triển v2 đạt precision
{pct(b2['test']['precision'])}, recall {pct(b2['test']['recall'])}. Chưa có kiểm định
độc lập tại Đà Nẵng; không gọi các số này là độ chính xác sử dụng ngoài thực tế.

## 1. Detector rác đa nguồn

- YOLO26n, seed 42, ảnh 640, batch 8, workers 0; thực chạy **{epoch_count} epoch**.
- Thời gian train: **{summary['train_seconds']/60:.2f} phút**. Best model chọn bằng
  validation; test chỉ dùng báo cáo sau khi khóa checkpoint.
- Dữ liệu: **1.655 train / 318 validation / 15 test**. WaRP: 1.000/200; HF TACO:
  600/100; Street negative: 55/3; giữ 15 val + 15 test TACO cũ. TACO mirror không
  phải nguồn độc lập với TACO gốc. 522 ảnh official test WaRP chưa dùng.
- Hash/dHash chống trùng là heuristic; duyệt nhãn thủ công vẫn pending. Tách WaRP
  theo ngày chụp và TACO theo batch nguồn; không có bộ test địa phương Đà Nẵng.

So sánh hai checkpoint trên **cùng** từng tập, không so các aggregate khác nhau:

| Tập | Pilot cũ mAP50 | Mới mAP50 | Thay đổi | Mới mAP50–95 | Mới recall |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

mAP là metric phát hiện box, không phải accuracy phân loại bảy vật liệu.
Test chỉ 15 ảnh nên chưa đại diện ngoài thực tế. Negative diagnostic: model mới
có detection ở {negatives['images_with_detection']}/{negatives['images']} ảnh (conf 0,25);
cỡ mẫu này không đủ kết luận false-positive rate.

Từng nhóm trên validation 318 ảnh:

| Nhóm vật thể | Precision | Recall | AP50 |
|---|---:|---:|---:|
{per_class}

Từng lớp lấy từ lượt validation sau train (batch 8); bảng so sánh domain dùng
batch 4. Sai khác số học nhỏ giữa hai lượt được giữ nguyên trong JSON nguồn.

Thời gian suy luận thử nghiệm GPU: median {summary['pilot_latency']['median_ms']:.2f} ms,
p95 {summary['pilot_latency']['p95_ms']:.2f} ms, 30 lần batch 1 gồm đọc ảnh, sau warmup.
Đây không phải benchmark thiết bị đích. Chưa tích hợp detector này thay luồng
MobileNet/SSDLite sản phẩm; chín nhóm không bao gồm pin hay nhãn medical.

## 2. Sàng lọc hình dạng pin

YOLO26n-cls hai nhãn `battery`, `not_battery`, **{b['epochs_completed']} epoch**,
224 px, batch 16. Nguồn Kaggle Garbage Classification. Sau lọc gần trùng:
{b['image_counts']['train']} train / {b['image_counts']['val']} validation /
{b['image_counts']['test']} test. Không có ID vật/cảnh để bảo đảm tách theo vật thật.

Ngưỡng **{b['threshold']:.2f}** chọn từ validation, không chọn theo test. Mục tiêu
validation recall ≥90%, precision ≥80%: **{'đạt trong miền' if b['validation_target_met'] else 'chưa đạt'}**.

| Tập | TP | FP | FN | TN | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | {validation['tp']} | {validation['fp']} | {validation['fn']} | {validation['tn']} | {pct(validation['precision'])} | {pct(validation['recall'])} | {pct(validation['accuracy'])} |
| Test nội bộ | {test['tp']} | {test['fp']} | {test['fn']} | {test['tn']} | {pct(test['precision'])} | {pct(test['recall'])} | {pct(test['accuracy'])} |

Kiểm tra ngoài miền với RecyBat24 v3 **chỉ có ảnh pin**: nhận dương
{external['screen_positive']}/{external['images']}, bỏ sót ở mức ảnh
{external['screen_negative']}/{external['images']}; recall ảnh **{pct(external['image_recall'])}**.
Không có ảnh negative nên **không đo được precision, specificity hoặc FPR** ở đây.
Nhiều ảnh là các mặt/ánh sáng của cùng viên pin, không coi là các vật độc lập.
Không tinh chỉnh model/ngưỡng theo tập này. Chưa kiểm tra pin ẩn hoặc ảnh Đà Nẵng.

Đã thêm API và tab **Pin và rác có nguy cơ** để dùng classifier thật trên ảnh cận
cảnh một vật. Không có bounding box pin; không xác định hóa học, hư hỏng, nguy cơ
cháy, nhiễm bệnh hoặc bức xạ. Score chưa hiệu chỉnh; âm tính không xác nhận an toàn.

### Bản cải tiến v2 sau thất bại ngoài miền

V1 bị loại khỏi runtime mặc định sau kết quả ngoài miền 0%; không hạ ngưỡng theo
tập chẩn đoán. V2 train lại từ pretrained với Kaggle + RecyBat, bổ sung negative
Kaggle chưa dùng vào train và cân bằng hai nhãn trong train. RecyBat tách theo
**110 ID viên pin (kèm dạng pin)**, gộp tất cả mặt/ánh sáng của cùng viên vào một
split, tối đa tám ảnh mỗi viên ở train. Các split Kaggle cũ được giữ nguyên.

V2: {b2['epochs_completed']} epoch; {b2['image_counts']['train']} train /
{b2['image_counts']['val']} validation / {b2['image_counts']['test']} development
holdout. Ngưỡng {b2['threshold']:.2f} chọn trên validation với mục tiêu recall ≥95%,
precision ≥90%: **{'đạt trong tập phát triển' if b2['validation_target_met'] else 'chưa đạt'}**.

Mục tiêu này tính trên tập gộp. **Chưa đạt recall ≥95% cho từng nguồn**:
Kaggle validation chỉ {pct(b2['per_source_validation']['kaggle_garbage']['recall'])}.
Không dùng nhiều ảnh pin RecyBat để che bỏ sót pin Kaggle.

| Tập v2 | TP | FP | FN | TN | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | {b2['validation']['tp']} | {b2['validation']['fp']} | {b2['validation']['fn']} | {b2['validation']['tn']} | {pct(b2['validation']['precision'])} | {pct(b2['validation']['recall'])} | {pct(b2['validation']['accuracy'])} |
| Development holdout | {b2['test']['tp']} | {b2['test']['fp']} | {b2['test']['fn']} | {b2['test']['tn']} | {pct(b2['test']['precision'])} | {pct(b2['test']['recall'])} | {pct(b2['test']['accuracy'])} |

Trên phần Kaggle của holdout v2: precision {pct(b2['per_source_test']['kaggle_garbage']['precision'])},
recall {pct(b2['per_source_test']['kaggle_garbage']['recall'])}. Trên ảnh pin RecyBat
holdout v2: recall {pct(b2['per_source_test']['recybat24_v3']['recall'])}; phần nguồn này
chỉ có positive nên precision ở đây không dùng kết luận năng lực báo nhầm.

**Đây là kiểm tra phát triển, chưa phải kiểm định ngoài miền độc lập**: RecyBat
đã được xem trong chẩn đoán v1 rồi được chọn làm nguồn phát triển v2. Không dùng
sự cải thiện để tuyên bố hiệu quả trên ảnh Đà Nẵng. Vẫn thiếu negative băng chuyền
độc lập, ID vật Kaggle và bộ ảnh địa phương. Trạng thái runtime: `{b2.get('runtime_status','experimental_development_only')}`;
chỉ phục vụ thử nghiệm có giám sát khi vượt mục tiêu validation.

## 3. Hướng dẫn xử lý và địa điểm

- Hướng dẫn có nguồn, tách lựa chọn của người dùng với kết quả thị giác. Pin nóng/
  bốc khói không được hướng dẫn đóng gói/vận chuyển tới điểm pin công cộng.
- Kim tiêm/rác y tế: cảnh báo phòng ngừa và liên hệ chuyên trách; chưa train model
  nhận diện medical mới. Không chẩn đoán bệnh cụ thể từ ảnh.
- Nghi phóng xạ: giữ tránh xa, báo cơ quan chức năng; cần thiết bị đo/chuyên môn.
  Không đo hay loại trừ bức xạ từ RGB, không có hướng dẫn xử lý phóng xạ tại nhà.
- Hướng dẫn riêng cho giấy, nhựa, lon/kim loại, chai lọ, hữu cơ, bao bì hỗn hợp.
- FEC từng công bố 108 Phan Châu Trinh và 808 Ngô Quyền, Đà Nẵng. Cả hai ghi
  **cần xác nhận hiện trạng**, không có bảo đảm đang nhận, giờ mở cửa, pin hỏng
  hoặc rác y tế/phóng xạ. Danh mục địa điểm cũ cũng ghi chưa xác minh.

Nguồn và kế hoạch nghiệm thu chi tiết: [Lộ trình Đà Nẵng](LO_TRINH_DANANG.md).

## 4. Kiểm thử phần mềm và cách chạy

**{passed} test độc lập đã qua**, thêm 4 subtest dữ liệu. Build frontend và lint
trả mã 0; lint còn cảnh báo cũ. `pip check` không có xung đột.

Kết quả từng lệnh, cảnh báo lint và lỗi suite đầy đủ được lưu nguyên văn trong
`MODEL_SCORECARD.json` và thư mục `verification/`. Không đổi các lỗi cũ thành pass.
Suite đầy đủ có lỗi kỳ vọng health `ok` dù thiếu MobileNet, yêu cầu HTTP 400 thay
503 khi model chưa sẵn sàng, và so khớp nguyên văn disclaimer. Xem log cuối để
biết số pass/fail thực tế; các benchmark cần checkpoint/ảnh local còn hạn chế.

Chạy từ `D:\\ai-waste-classification`:

```powershell
backend\\.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
npm.cmd --prefix frontend run dev -- --host 127.0.0.1
```

Mở http://127.0.0.1:5173, chọn tab **Pin và rác có nguy cơ**. Health MobileNet
vẫn `degraded` do thiếu `mobilenetv3_best.pth`; dùng `/api/safety/capabilities` và
thử suy luận thực để kiểm tra classifier pin. Hai năng lực được báo riêng.

Artifact đã chép vào `waste_detector/` và `battery_screen/`, gồm weights, báo cáo,
môi trường và hash. Dữ liệu/weights vẫn ngoài Git. Để dùng bản sao pin, đặt
`BATTERY_SCREEN_RUN` tới thư mục `battery_screen` có report và `weights/best.pt`.

## 5. Bước tiếp theo

Ưu tiên duyệt nhãn và thu thập test Đà Nẵng, negative khó, pin cúc áo/che khuất,
ảnh nhiều vật; sau đó train detector pin từ box gốc RecyBat + ảnh địa phương.
Không tiếp tục tăng dữ liệu không tương thích. SAM có thể hỗ trợ gán nhãn sau
khi người duyệt xác nhận, chưa được chạy trong lượt này. Khi mở rộng Bắc/Nam,
phải có test theo miền và registry điểm nhận được xác minh.

**Chưa đạt nghiệm thu dùng không giám sát.** Cần cải thiện recall, kiểm định
ngoài miền, chuyên gia duyệt hướng dẫn và xác nhận điểm xử lý trước triển khai.
'''
    (ROOT/'docs/MULTISOURCE_BATTERY_RESULTS.md').write_text(report.replace('](LO_TRINH_DANANG.md)', '](HAZARD_ROADMAP_DANANG.md)'),encoding='utf-8')
    (output/'BAO_CAO_TRAIN_DANANG_2026-11-10.md').write_text(report,encoding='utf-8')
    (output/'MODEL_SCORECARD.json').write_text(json.dumps(scorecard,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(ROOT/'docs/HAZARD_ROADMAP_DANANG.md',output/'LO_TRINH_DANANG.md')
    shutil.copy2(ROOT/'docs/DANANG_CAPTURE_PROTOCOL.md',output/'THU_THAP_ANH_DANANG.md')
    for source,folder,files in [(detector,'waste_detector',['weights/best.pt','pilot_summary.json','comparison.json','results.csv','environment.txt']),
                                (battery,'battery_baseline_v1',['weights/best.pt','battery_report.json','external_recybat_report.json','results.csv','environment.txt']),
                                (battery_v2,'battery_screen',['weights/best.pt','battery_report.json','results.csv','environment.txt'])]:
        for name in files:
            target=output/folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    verification=output/'verification';verification.mkdir(exist_ok=True)
    for name in scorecard['test_logs']:shutil.copy2(ROOT/'logs'/name,verification/name)
    for dataset in ['waste_multisource_v3','battery_screen_v1','battery_multisource_v2']:
        for name in ['manifest.json','audit.json','audit_checks.json','dataset.yaml','warp_mapping.json']:
            source=ROOT/'data/processed'/dataset/name
            if source.is_file():
                target=output/'dataset_audits'/dataset/name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for name in ['api_safety_smoke.json','battery_api_smoke.json','battery_api_baseline_v1_smoke.json','battery_external_pipeline_check.json']:
        if (ROOT/'logs'/name).is_file():shutil.copy2(ROOT/'logs'/name,verification/name)
    print('REPORT_COMPLETE',output)


if __name__=='__main__':main()
