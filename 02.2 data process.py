# save as create_data_yaml.py

data_yaml_content = """
path: ./ppe_human_detection  # dataset root directory

train: images/train
val: images/val
test: images/test

names:
  0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",        
    6: "person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots"
"""

with open("ppe_human_detection/data.yaml", "w") as f:
    f.write(data_yaml_content.strip())

print("✅ data.yaml with 11 classes created!")
