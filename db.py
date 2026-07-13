import os
import json

naked_dir = r"E:\AI-Toolkit\AI-Toolkit-XPU-zh_cn\datasets\anime"
jsonl_path = r"putong_tag_result.jsonl"

# 加载映射
name_to_desc = {}
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        name_to_desc[data["filename"]] = data["description"]

# 写入txt
for img_name, desc in name_to_desc.items():
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_full = os.path.join(naked_dir, txt_name)
    with open(txt_full, "w", encoding="utf-8") as f:
        f.write(desc)
print("全部jsonl描述已生成对应txt文件")