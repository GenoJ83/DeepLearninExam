CSC3218 submission package — Geno Owor Joshua (M23B23/006)

Contents:
  code/           train.py, data.py, model.py, requirements.txt
  code/generate_presentation.py — from this folder run: python code/generate_presentation.py
  report/         report_template.tex (+ ucu_logo.png if present; add compiled PDF here)
  results/        metrics.json, classification_report.txt, figures, best_model.pt
  CSC3218_presentation.pptx
  CSC3218_CIFAR10_workflow.ipynb (standalone Colab-style notebook)

Before submitting: compile report/report_template.tex to PDF (e.g. pdflatex x2) and place
the PDF in report/ if your instructor requires a PDF.

Re-run training from this folder (example):
  cd code && python train.py --data-dir ../data --out-dir ../results
  cd .. && python code/generate_presentation.py

Note: CIFAR-10 raw data (data/) is not included — torchvision downloads it when you run train.py.
