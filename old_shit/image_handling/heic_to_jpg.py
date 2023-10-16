import os
from wand.image import Image

src_dir = r'I:\.shortcut-targets-by-id\1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs\Åsalena\heic'
dst_dir = r'I:\.shortcut-targets-by-id\1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs\Åsalena\jpg'

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.lower().endswith('.heic'):
            full_file_path = os.path.join(root, file)
            with Image(filename=full_file_path) as img:
                relative_path = os.path.relpath(root, src_dir)
                dst_folder = os.path.join(dst_dir, relative_path)
                os.makedirs(dst_folder, exist_ok=True)
                dst_file_path = os.path.join(dst_folder, os.path.splitext(file)[0] + '.jpg')
                img.save(filename=dst_file_path)

print("Conversion completed!")
