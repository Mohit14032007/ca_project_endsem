# assemble_report.py: Assembles full markdown report and verifies invariants

import os, ast

parts = [
    "mohman/final_report/builder/part1.md",
    "mohman/final_report/builder/part2.md",
    "mohman/final_report/builder/part3.md",
    "mohman/final_report/builder/part4.md"
]

full_content = []
for p in parts:
    with open(p, "r", encoding="utf-8") as f:
        full_content.append(f.read())

combined_text = "\n\n".join(full_content)

out_path = "mohman/final_report/MoE_Memory_Wall_Detailed_Report.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(combined_text)

print(f"Master Markdown Report written to {out_path}")
print(f"Total lines: {len(combined_text.splitlines())}")
print(f"Total characters: {len(combined_text)}")

# Verify zero ternary operators in builder scripts
builder_scripts = [
    "mohman/final_report/builder/build_part1.py",
    "mohman/final_report/builder/build_part2.py",
    "mohman/final_report/builder/build_part3.py",
    "mohman/final_report/builder/build_part4.py",
    "mohman/final_report/builder/assemble_report.py"
]

for s in builder_scripts:
    with open(s, "r") as f:
        tree = ast.parse(f.read(), filename=s)
    ternary_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.IfExp)]
    print(f"Script {s}: {len(ternary_nodes)} ternary operators found.")
    assert len(ternary_nodes) == 0, f"Violation: Ternary operator found in {s}!"

print("ALL INVARIANTS SATISFIED: Zero ternary operators verified.")
