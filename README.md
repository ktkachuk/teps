# TEPS: Tool Engagement based Phase Segmentation 

**TEPS** is a lightweight, unsupervised algorithm for real-time monitoring and segmentation of drilling operations in CNC machine tools.

This repository provides an easy-to-use Python implementation of the algorithm, with an animated visualization of the segmentation process.

![TEPS Demo](utils/teps_demo_animation.gif)


---

## How to Run

To try the algorithm and see it in action:

```bash
python example.py
```

This runs the example that:
- Initializes the TEPS model
- Streams and processes the input signal
- Displays a real-time animated plot of the drilling signal, with color-coded phase segmentation

---

## About the Method

Real-time monitoring of drilling operations in Computer Numerical Control (CNC) machine tools is crucial to ensure process reliability and to minimize downtime. This is enabled by the detection of anomalies such as missing workpieces, incorrect measurements, and drill bit breakages.

The TEPS algorithm works by segmenting the drilling process into distinct phases and identifying unexpected transitions. The segmented data can then support advanced monitoring, analytics, and process optimization.

This approach:
- Requires no labeled training data
- Adapts online in real time
- Is inspired by k-means clustering
- Can be run directly on raw streaming data from the CNC-control unit
- Requires minimal memory and computation

---

##  Citation

If you use this code or method in your work, **please cite the original publication**:

> **Real-Time Phase Segmentation for CNC Drilling:** 
>
> A Lightweight Unsupervised Machine Learning Approach for Condition Monitoring and Anomaly Detection
> 
> Kirill Tkachuk , Berend Denkena, Henning Buhl
>  
**Please cite this work if you use it.**
