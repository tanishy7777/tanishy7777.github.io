---
layout: post
title: Graham Scan Finding the Convex Hull in Object Detection
date: 2025-02-10 22:02:00 +5:30
description: Learn how the Graham Scan algorithm works to find the convex hull of keypoints in object detection with fun explanations and math
tags: algorithms
categories: algorithms
thumbnail: assets/img/9.jpg
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# Introduction  
Hey there! Ever wondered how we can wrap a tight boundary around a bunch of points, like tracing the edges of a cloud of stars in the night sky? That’s essentially what the **Graham Scan algorithm** does—it finds the smallest convex polygon that contains all the given points, called the **convex hull**.

In object detection, this is super useful for finding **bounding boxes** that are more accurate than simple rectangular boxes. So, let’s dive into how it works!

---

## What is a Convex Hull?  
Imagine placing a rubber band around a set of nails on a board. When you release it, the rubber band snaps to form a tight boundary around the nails. This boundary is the convex hull—a polygon where every interior angle is less than 180 degrees.

In mathematical terms, given a set of points $ P = \{p_1, p_2, \dots, p_n\} $, the convex hull is the smallest convex polygon that encloses all the points.

---

## The Graham Scan Algorithm  
The Graham Scan algorithm finds the convex hull in **$O(n \log n)$** time. Here’s the step-by-step breakdown:

1. **Find the point with the lowest y-coordinate** (break ties by choosing the leftmost point). This will be our starting point $ p_0 $.
2. **Sort the remaining points by polar angle** relative to $ p_0 $.  
3. **Traverse the sorted points and construct the hull** using a stack:
   - Push the first two points onto the stack.
   - For each subsequent point:
     - While the angle formed by the top two points and the current point makes a **right turn**, pop the top point from the stack.
     - Push the current point onto the stack.
4. The points remaining in the stack form the convex hull.

---

## Math Behind It  
To determine if three points $ p_1 $, $ p_2 $, and $ p_3 $ make a left or right turn, we use the **cross product** of the vectors:  

$$
\text{Cross Product} = (x_2 - x_1)(y_3 - y_1) - (y_2 - y_1)(x_3 - x_1)
$$

- If the result is **positive**, the points make a **left turn**.
- If it’s **negative**, they make a **right turn**.
- If it’s **zero**, the points are collinear.

---

## Why Use This in Object Detection?  
In object detection, keypoints often define the boundaries of an object. Using a convex hull instead of a simple bounding box can provide a tighter, more accurate boundary around irregularly shaped objects. This can be crucial for applications like:
- Autonomous driving (detecting pedestrians and vehicles)
- Medical imaging (outlining tumors or organs)
- Robotics (object grasping and manipulation)

---

## Python Implementation
Here’s a Python snippet to implement the Graham Scan algorithm:

```python
def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def graham_scan(points):
    points = sorted(points)
    lower, upper = [], []

    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

# Example usage
points = [(0, 0), (1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)]
hull = graham_scan(points)
print("Convex Hull:", hull)
