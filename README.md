# pytorch-debugging-checklist
Practical PyTorch debugging checklist when your neural net won't learn. Includes data sanity checks, broken baseline, tiny-subset overfit test, LR finder, Kaiming init, AdamW+OneCycleLR training, and final debug dashboard with confusion matrix. MNIST example reaches ~99.1% accuracy.

# Neural Network Not Learning? The Ultimate Debugging Checklist

> "I've spent embarrassing amounts of time staring at a flat loss curve, convinced my GPU was broken, only to realize I forgot `model.train()`."

A practical, 7-step debugging checklist for PyTorch neural networks that actually works.

### Author
**Emmimal P Alexander**

### Original Article
Read the full blog post here:  
🔗 [https://emitechlogic.com/neural-network-not-learning-a-7-step-debugging-checklist-that-actually-works/](https://emitechlogic.com/neural-network-not-learning-a-7-step-debugging-checklist-that-actually-works/)

### Features
- Check 1: Data Pipeline Sanity  
- Check 2: Broken Baseline (what failure looks like)  
- Check 3: Overfit Tiny Subset (50 samples)  
- Check 4: Learning Rate Finder  
- Check 5: Proper Weight Initialization (Kaiming)  
- Check 6: Full Training (AdamW + OneCycleLR)  
- Check 7: Final Debug Dashboard (confusion matrix + per-class accuracy)

### Quick Start

```bash
git clone https://github.com/Emmimal/pytorch-debugging-checklist.git
cd pytorch-debugging-checklist
pip install -r requirements.txt
python debugging_checklist.py

### Requirements

- Python 3.8+
- PyTorch 2.x
- torchvision
- matplotlib

### How to Cite This Work

If you use this checklist or code in your research, project, or article, please cite it as:
```
@misc{alexander2026pytorchdebug,
  author       = {Emmimal P Alexander},
  title        = {Neural Network Not Learning? The Ultimate Debugging Checklist},
  year         = {2026},
  howpublished = {\url{https://emitechlogic.com/neural-network-not-learning-a-7-step-debugging-checklist-that-actually-works/}},
  note         = {GitHub Repository: pytorch-debugging-checklist}
}
```

### APA Style:
Emmimal P. Alexander. (2026). Neural Network Not Learning? The Ultimate Debugging Checklist. https://emitechlogic.com/neural-network-not-learning-a-7-step-debugging-checklist-that-actually-works/
