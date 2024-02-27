import time

dir = "actual_project"
file_name = "body_and_face_detection.py"
file_path = f"{dir}/{file_name}"
code_to_analyze = open(file_path, 'r').readlines()

code = '''import time
import sys
lines_times = {}'''
code += "\nsys.path.append(\""+dir+"\")"
lines = [line.replace("\n", "") for line in code_to_analyze]
for i, line in enumerate(lines):
    line = line.replace("\"", "\\\"")
    code += "\nlines_times[\""+line+"\"] = { \"total_time\": 0, \"times\": 0, \"linenumber\": \""+str(i)+"\" }"

incorrect_starts = ["#", "for", "with", "def", "if", "return", "while", "break", "continue", "else", "elif", "try", "except", "finally", "import", "from"]
for line in lines:
    if line.strip() == "":
        continue
    lines_to_add = []
    is_incorrect_start = False
    for incorrect_start in incorrect_starts:
        if line.strip().startswith(incorrect_start):
            is_incorrect_start = True
            break
            
    if is_incorrect_start:
        code += "\n" + line
        continue

    indentation = 0
    for char in line:
        if char == " ":
            indentation += 1
        else:
            break
        
    lines_to_add.append("start_time_temp = time.time()")
    lines_to_add.append(line)
    lines_to_add.append("time_for_line = time.time() - start_time_temp")
    line_key = line.replace("\"", "\\\"")
    lines_to_add.append("lines_times[\""+line_key+"\"][\"total_time\"] += time_for_line")
    lines_to_add.append("lines_times[\""+line_key+"\"][\"times\"] += 1")
    correct_indentation = " " * indentation
    for i, line_to_add in enumerate(lines_to_add):
        if i == 1:
            continue
        lines_to_add[i] = correct_indentation + line_to_add
    code += "\n"+"\n".join(lines_to_add)

code += """
total_time = sum([data['total_time'] for data in lines_times.values()])
for line, data in lines_times.items():
    if data['times'] > 0:
        print("{:>4} {:<100} --- tottime: {:<5}, times: {:<4}, percent: {:<4}, avg: {:<4}".format(data['linenumber'], line, "{:.2f}".format(data['total_time']), data['times'], "{:.2f}".format(data['total_time'] / total_time * 100), "{:.2f}".format(data['total_time'] / data['times'])))
    else:
        print("{:>4} {:<100} --- tottime: {:<5}, times: {:<4}, percent: {:<4}".format(data['linenumber'], line, "{:.2f}".format(data['total_time']), data['times'], "{:.2f}".format(data['total_time'] / total_time * 100)))
print("\\n\\n\\n\\n")
sorted_lines = sorted(lines_times.items(), key=lambda x: x[1]['total_time'], reverse=True)
for line, data in sorted_lines:
    print("{:>4} {:<100} --- tottime: {:<5}, times: {:<4}, percent: {:<4}".format(data['linenumber'], line, "{:.2f}".format(data['total_time']), data['times'], "{:.2f}".format(data['total_time'] / total_time * 100)))
"""

exec(code)

