# DL-017: Object Detection with a Pretrained YOLO Model

A beginner-friendly deep learning practical. **No training is needed** - we reuse a model
that someone else already trained (that is what "pretrained" means) and study how it behaves.

## 1. Concepts you need (read this before the viva)

| Term | Plain meaning |
|---|---|
| Classification | Looks at a whole image and answers "what is in it?" (one label) |
| Object detection | Answers "what is in it AND where?" - a box around each object |
| Bounding box | Rectangle `(x1, y1, x2, y2)` around an object |
| YOLO | "You Only Look Once" - a fast detector that finds all boxes in a single pass of the neural network |
| Pretrained weights | The numbers a network learned earlier. We load them instead of training |
| COCO dataset | The dataset YOLOv8 was trained on: 80 everyday classes (person, car, bus, dog, chair, laptop...) |
| Inference | Using a trained model to make predictions on new images |
| Confidence score | Number from 0 to 1: how sure the model is about a box. It is NOT the probability of being correct |
| Confidence threshold | Cut-off. Boxes scoring below it are thrown away |
| False positive | Model draws a box where there is no such object (or wrong label) |
| False negative | An object is present but the model missed it |

**Threshold trade-off:** low threshold -> more boxes, more false positives, fewer misses.
High threshold -> fewer boxes, cleaner output, but more missed objects.

## 2. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Needs internet on the first run (downloads `yolov8n.pt`, about 6 MB). First install is large because it pulls PyTorch.
No GPU required. Tip: if your laptop is slow or has no internet, use Google Colab.

## 3. Run

```bash
python detect.py --source images
python detect.py --source images --conf 0.1 0.3 0.5 0.7     # your own thresholds
python detect.py --source my_video.mp4                       # short video (<= 150 frames by default)
```

Put 10-20 of your own non-sensitive images (street, classroom objects, kitchen, desk) in `images/`.
Two sample images (`bus.jpg`, `zidane.jpg`) are included from the Ultralytics repo.

## 4. What you get in `outputs/`

- `annotated/conf_0.10/ ... conf_0.75/` - images with boxes, labels, scores
- `detections.csv` - every detection (image, class, confidence, box coordinates)
- `threshold_summary.png` - chart of detections and average confidence per threshold
- `compare_<image>.png` - same image at every threshold, side by side
- `<video>_detected.mp4` - annotated video (if you gave a video)

## 5. How requirements map to the code

| Requirement | Where |
|---|---|
| Load pretrained weights | `YOLO(args.model)` in `main()` |
| Run inference | `model.predict(...)` in `run_on_image()` / `process_video()` |
| Tune confidence threshold | `--conf` list, loop in `process_images()` |
| Visualize boxes and labels | `result.plot()`, `plot_comparison()` |
| Annotated outputs with scores | saved images + `detections.csv` |
| Review errors | fill the `review (TP/FP)` column in the CSV + missed objects table below |
| OpenCV / Matplotlib | image + video I/O, colour conversion / charts |

## 6. Error review (do this by hand - it is the most valuable part of the report)

Open the annotated images at 0.25 and look carefully. Fill in a table like:

| Image | Problem | Type | Possible reason |
|---|---|---|---|
| img03.jpg | Backpack labelled "handbag" | Wrong label | Similar shape |
| img07.jpg | Small cup on far desk not detected | Missed object (FN) | Object too small/blurry |
| img09.jpg | Chair detected on a sofa | False positive | Overlapping similar objects |

Then count TP / FP per threshold using the CSV and write a 3-4 line conclusion about which threshold you would pick.

## 7. Suggested report outline

1. Problem statement and motivation (classification vs detection)
2. How YOLO works (one paragraph + the terms table above)
3. Dataset: where images came from (self-captured or public, non-sensitive)
4. Method: model, weights, thresholds, tools
5. Results: annotated examples, threshold chart, comparison figure
6. Error analysis: false detections and missed objects
7. Limitations and ethics: small sample, COCO classes only, lighting/size sensitivity, not for surveillance or high-stakes use
8. Conclusion
9. References

## 8. Citation and license (required)

- Model: Ultralytics YOLOv8 - Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLO*. https://github.com/ultralytics/ultralytics - licensed **AGPL-3.0**.
- Original YOLO paper: Redmon et al. (2016), "You Only Look Once: Unified, Real-Time Object Detection", CVPR.
- Pretrained dataset: Lin et al. (2014), "Microsoft COCO: Common Objects in Context".
- Sample images: from the Ultralytics repository assets. Replace with your own for the final submission.
- Check the licence of any public images you add and cite their source.

## 9. Likely viva questions

1. Difference between classification and detection? 2. What does YOLO stand for and why is it fast?
3. What is a confidence score? Does 0.9 mean it is 90% correct? (No.) 4. What happens when you raise the threshold?
5. What is a false positive vs false negative? 6. Why did we not train the model? (Transfer of existing learning; pretrained on COCO.)
7. Why can the model miss small objects? 8. Why shouldn't this be used for surveillance or critical decisions?

## 10. Optional extensions (only if time permits)

Compare `yolov8n.pt` vs `yolov8s.pt` (bigger, slower, more accurate); filter only some classes (`classes=[0, 2]` in `predict`); count objects per image.
