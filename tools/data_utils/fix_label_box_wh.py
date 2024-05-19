from glob import glob
from tqdm import tqdm


def fix_boxes(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        box = line.strip('\n').split(' ')
        fix_dx = box[4]
        fix_dy = box[3]
        box[3] = fix_dx
        box[4] = fix_dy
        new_line = " ".join(box) + "\n"
        boxes.append(new_line)
    dst_label = filepath.replace("labels_old", "labels")
    with open(dst_label, "w", encoding="utf-8") as file:
        file.writelines(boxes)

old_labels = glob(f"/hy-tmp/data/custom/labels_old/*.txt")
for label in tqdm(old_labels):
    fix_boxes(label)