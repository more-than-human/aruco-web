# ArUco Nano: a simpler, faster, and more reliable fiducial marker detector

**Sergio Garrido-Jurado** <sup>a</sup>, **Francisco J. Romero-Ramirez** <sup>b</sup>, **Rafael Muñoz-Salinas** <sup>a,∗</sup>

<sup>a</sup> _Department of Computer Science and Artificial Intelligence, Córdoba University, Córdoba, Andalusia, Spain_  
<sup>b</sup> _Department of Electronics and Computer Engineering, Córdoba University, Córdoba, Andalusia, Spain_

---

## ARTICLE INFO

**Keywords:**  
ArUco  
Fiducial markers  
Camera localization  
Robotics

## ABSTRACT

Squared fiducial markers are essential for many augmented reality and robotics applications, with ArUco serving as the _de facto_ standard, largely due to its integration with OpenCV. However, the standard implementation relies on general-purpose algorithms that are not specifically tailored to this task, resulting in significant computational bottlenecks, especially on resource-constrained devices. We introduce ArUco Nano, a minimalist (500 lines of code), header-only C++ library that achieves a speedup of up to 6.5× over the standard OpenCV library. By implementing a novel Visited-Aware contour extraction algorithm and direct sub-pixel code sampling, the library avoids computationally expensive contour extraction and image warping processes while also improving detection reliability. Benchmarks on up to 16 MP imagery demonstrate that ArUco Nano consistently outperforms the current OpenCV implementation in our evaluation, achieving higher speed and a higher F1 score.

---

## Metadata

### Code metadata.

| Nr. | Code metadata description                                       | Metadata                                |
| :-- | :-------------------------------------------------------------- | :-------------------------------------- |
| C1  | Current code version                                            | 8                                       |
| C2  | Permanent Link to code/repository used for this code version    | https://github.com/rmsalinas/aruco_nano |
| C3  | Permanent link to Reproducible Capsule                          | n/a                                     |
| C4  | Legal Code License                                              | MIT License                             |
| C5  | Code versioning system used                                     | git                                     |
| C6  | Software code languages, tools, and services used               | C++, cmake                              |
| C7  | Compilation requirements, operating environments & dependencies | OpenCV $\ge$ 4.7                        |
| C8  | If available Link to developer documentation/manual             | n/a                                     |
| C9  | Support email for questions                                     | rmsalinas@uco.es                        |

---

## 1. Motivation and significance

Squared fiducial markers are artificial landmarks that serve as unique visual identifiers within a scene. By encoding a binary ID into a known geometric pattern, they allow computer vision algorithms to unambiguously identify objects and estimate their 3D pose with high precision. Due to this reliability, they have become the functional backbone for diverse high-stakes applications: from autonomous warehouse logistics and industrial automation [1–4] to robot navigation [5–9], medical assistance [10,11], and consumer-grade augmented reality [12–14].

Among existing systems, ArUco has emerged as a standard, in part because of its superior performance [15] and its integration into the OpenCV library [16], one of the most widely used computer vision libraries. The impact of the ArUco framework is profound, as evidenced by the extensive citations of its foundational works [17–19]. However, the current ArUco implementation relies on standard image-processing algorithms in the OpenCV library, which are designed for general-purpose use rather than marker detection. This introduces significant computational overhead, including steps such as the Suzuki-Abe algorithm [20] for contour extraction and full-image warping for marker identification, which can become bottlenecks in resource-constrained environments.

In this context, we introduce **ArUco Nano**, a minimalist, header-only C++ library for fiducial marker detection. By implementing a novel Visited-Aware contour extraction and direct sub-pixel code sampling, ArUco Nano achieves a speedup of up to 6.5× over the standard ArUco library. Given ArUco’s widespread adoption across industries worldwide, such optimization directly translates into higher detection rates and lower power consumption, enabling computer-vision workflows to run on significantly cheaper, less powerful hardware.

---

## 2. Software description

### 2.1. Software architecture

![Overview of ArUco Nano pipeline](fig1.png)  
_Fig. 1. Overview of ArUco Nano pipeline for marker detection and pose estimation. (The diagram shows an input image and detection parameters passed to the Marker Detection stage, which consists of Image Segmentation, Contour Extraction, and Marker Identification. This outputs detected markers with their IDs and 2D corners. Optionally, the 3D Pose Estimation block takes the detected markers along with marker size and intrinsic parameters to output the 3D pose, represented as rotation and translation vectors)._

Fig. 1 provides a high-level overview of the library’s pipeline. Marker detection is performed by the `aruco_nano::MarkerDetector` class through the `aruco_nano::MarkerDetector::detect()` method, which takes an input image in OpenCV format (`cv::Mat`) and returns the detected markers as `std::vector<aruco_nano::Marker>`. Each marker is defined by its numeric identifier and the image coordinates of its four corners. Detection can be configured using `aruco_nano::Params`.

Optionally, the 3D pose of a detected marker can be estimated using `aruco_nano::Marker::estimatePose()`, which takes the camera intrinsic matrix and the distortion coefficients as `cv::Mat` in the standard OpenCV format, along with the marker’s physical size specified as the length of one side, and returns rotation and translation vectors following the OpenCV convention (`cv::Mat rvec, tvec`).

In Section 2.2, we describe each block in more detail and discuss the specific implementation and optimizations introduced in ArUco Nano. Section 3 provides concrete usage examples.

### 2.2. Software functionalities

This section presents the main functionalities of ArUco Nano. The detection pipeline comprises image segmentation (Section 2.2.1), contour extraction (Section 2.2.2), and marker identification (Section 2.2.3). We then describe pose estimation in Section 2.2.4 and summarize the main configuration parameters in Section 2.2.5.

#### 2.2.1. Image segmentation

The first step segments the input image to highlight marker boundaries. If necessary, the image is first converted to grayscale. We then replace the computationally expensive `cv::adaptiveThreshold` used in the standard ArUco implementation with a lightweight high-pass filter: a box filter ($S_{\text{win}} = 15$) computes a local mean, which is subtracted from the original image. A low binary threshold ($\tau_{\text{diff}} = 3$) is then applied to the resulting difference map. This approach extracts edges while minimizing overhead, providing an $\sim 1.2\times$ speedup for this stage.

#### 2.2.2. Contour extraction

In this stage, we process the segmented image to extract candidate marker contours. This step constitutes the primary source of performance gain in ArUco Nano. The standard OpenCV implementation relies on `cv::findContours`, which employs the Suzuki-Abe algorithm [20] to recover the full contour set together with its complete topological hierarchy. For marker detection, this hierarchy is unnecessary and, more critically, spurious contours from noise often far outnumber valid marker candidates, introducing significant overhead in later stages.

To address this, we implement a custom _Visited-Aware Contour Tracing_ algorithm. The binary image is scanned in raster order (line by line) to identify potential marker borders using a novel contour-tracking algorithm inspired by Moore’s algorithm [21]. The substantial speedup achieved here is twofold: first, the intrinsic efficiency of our tracing implementation, and second, a substantial reduction in the workload of later stages by aggressively rejecting noisy contours and small blobs early.

Unlike standard Moore tracing, our method integrates a noise-rejection heuristic directly into the pixel-traversal loop. Since valid markers are convex quadrilaterals, their boundaries should be traced sequentially with minimal self-intersection. Noise artifacts, however, often exhibit jagged shapes, causing the tracer to revisit pixels frequently. To handle noise artifacts in small valid markers, we introduce a tolerance threshold $R_{\max}$. If the cumulative number of revisited pixels exceeds $R_{\max}$ during tracing, the contour is classified as noise and discarded, preventing further memory allocation. The proposed contour-tracing algorithm is presented in Algorithm 1 and illustrated in Fig. 2.

```
Algorithm 1: Visited-Aware Contour Tracing.
Input: Binary Image I of size H × W, Min Contour Size T_min, Max Revisits R_max
Output: List of Contours L
1  Initialize L ← ∅
2  for y ← 0 to H - 1 do
3    for x ← 0 to W - 2 do
4      p ← (y, x); p_right ← (y, x + 1)
5      if I(p) is Background and I(p_right) is Foreground then
6        // Start of a contour
7        C ← ∅; n_revisits ← 0
8        p_curr ← p; p_start ← p_right
9        repeat
10         I(p_curr) ← VISITED
11         C ← C ∪ {p_curr}
12         p_next ← MOORENEIGHBOR(p_curr)
13         if I(p_next) is Visited then
14           n_revisits ← n_revisits + 1
15         end
16         p_curr ← p_next
17       until p_curr = p_start
18       if n_revisits ≤ R_max and size(C) ≥ T_min then
19         Add C to L
20       end
21     end
22   end
23 end
```

![Visited-Aware contour tracing algorithm example](fig2.png)  
_Fig. 2. Visited-Aware contour tracing algorithm example. (a) A pixel grid diagram illustrating contour tracing on two shapes: an irregular non-marker shape on the left with multiple revisited pixels (red links, $n_{\text{revisits}} = 5$) and a rectangular marker shape on the right with minimal revisits ($n_{\text{revisits}} = 2$). (b) A real-world image patch example showing a noisy binary contour affected by an extra edge pixel at the top-right corner. (See text for details)._

Fig. 2(a) shows two contours: one that is clearly not a possible marker (the left one) and another one that could be (the right one). The figure shows how the algorithm starts tracing the left contour at pixel $(2, 2)$ and traverses it clockwise. As can be seen, most pixels are visited more than once (red links), having a total $n_{\text{revisits}} = 5$. The second contour shows a marker with some extra pixels due to noise. The contour following starts at pixel $(2,7)$, and it can be seen that pixel $(2,15)$ is visited twice. Therefore, the number of revisits at the end of the process will be two: one for pixel $(2, 15)$ and one for the starting pixel. The problem is better illustrated with a real example in Fig. 2(b), showing how the top right corner of the marker contour is affected by a noise pixel.

By discarding these artifacts at the tracing level, we minimize the number of candidates that require further processing. The resulting valid contours are then approximated to polygons using the Douglas-Peucker algorithm [22]. Only convex polygons with exactly 4 vertices that meet the minimum size constraints are retained as marker candidates.

#### 2.2.3. Marker identification

For each candidate, we must extract and verify its internal code to determine whether it corresponds to a valid marker. We first compute a homography that maps the observed quadrilateral to a canonical square. In the current ArUco implementation, this homography is used to warp the entire marker patch, which is then analyzed to recover the bit pattern. This full-image warping is costly, so we replace it with a direct sub-pixel sampling method.

Once the homography is estimated, rather than generating a warped marker image, we directly sample the center of each bit cell in the original image using an optimized sub-pixel interpolation routine, thus reducing the computational cost per candidate. The extracted bit pattern is then validated against the selected dictionary (e.g., `DICT_ARUCO_MIP_36h12`).

For small markers, sampling becomes more sensitive to pixel quantization, and the estimated bit centers may be slightly mislocalized and occasionally fall near the boundary of a neighboring cell, leading to identification failure. To improve robustness in these cases, we employ a stochastic refinement procedure: if the initial decoding fails, we apply small random perturbations, $\mathcal{N}(\mu = 0, \sigma = 0.75)$, to the detected corners, recompute the homography, and retry decoding for up to $K_{\max} = 5$ attempts. If decoding still fails after these attempts, the candidate is rejected.

Overall, this approach is not only faster than the original warping-based method but also improves marker identification in challenging cases through the stochastic refinement step, as shown in Section 4.

#### 2.2.4. Pose estimation

Once a marker is detected and its corners are identified in the image plane, ArUco Nano can estimate its 3D position and orientation relative to the camera. This process is handled by the `aruco_nano::Marker::estimatePose()` method, which internally implements a Perspective-n-Point (PnP) solver. To achieve high precision, we apply sub-pixel refinement to the detected corners before passing them to the solver, ensuring sub-pixel localization.

Pose estimation requires two sets of external data: the camera’s intrinsic parameters (camera matrix and distortion coefficients), obtained from a prior calibration, and the marker’s physical size (typically its side length in meters). The library then solves for the geometric transformation that maps the marker’s 3D corner coordinates in its local object space to the observed 2D image coordinates. The result is a rotation vector (`rvec`) and a translation vector (`tvec`) following the standard OpenCV convention. Together, they define the rigid transformation between the marker and the camera, which is fundamental for marker-based applications such as robotic navigation and augmented reality.

#### 2.2.5. Detection configuration

Marker detection parameters can be configured through the `aruco_nano::Params` class. However, the default parameters used in our experiments provide satisfactory results in most cases. Table 1 summarizes the mapping between the internal software parameters and the notation used in this paper.

- **Window Size ($S_{\text{win}}$):** Contrary to OpenCV’s default practice, which performs the adaptive threshold and contour extraction multiple times (three by default) on the original image using different window sizes, we found that larger windows consistently yield better boundary definitions in combination with low threshold values. In addition, due to the use of optimized separable filters and SIMD instructions in `cv::boxFilter`, the computational cost is largely independent of the kernel size. We default to a relatively large kernel ($S_{\text{win}} = 15$) to ensure robustness against lighting gradients without a performance penalty.
- **Threshold Constant ($\tau_{\text{diff}}$):** We use a low difference threshold (default $\tau_{\text{diff}} = 3$) to ensure that marker borders are detected even under low-contrast conditions. While this typically generates many noise contours, our Visited-Aware tracing algorithm (Section 2.2.2) efficiently discards these artifacts before they consume memory, making this aggressive thresholding feasible.
- **Minimum Contour Size ($L_{\min}$):** To reduce memory allocations, we enforce a minimum side length constraint during the tracing process ($L_{\min} = 10$ pixels), effectively rejecting any blob with a perimeter smaller than $T_{\min} = 4L_{\min}$.
- **Max Revisit Tolerance ($R_{\max}$):** While ideal convex shapes should not revisit pixels during contour tracing, discretization effects in small markers and noise can cause valid contours to have pixels revisited more than once, as shown in Fig. 2. We set a tolerance of $R_{\max} = 4$. Contours exceeding this count are classified as noise and rejected.
- **Refinement Attempts ($K_{\max}$):** For the stochastic corner refinement described in Section 2.2.3, we set the maximum number of perturbations to $K_{\max} = 5$. This provides a good trade-off between detection recovery and worst-case processing time.

The default values were empirically selected and found to be robust across a wide range of scenes. Moderate variations of $S_{\text{win}}$ and $\tau_{\text{diff}}$ did not significantly affect detection performance.

### Table 1: Default parameter configuration and notation.

| Software Parameter        | Symbol               | Description           | Default |
| :------------------------ | :------------------- | :-------------------- | :------ |
| `boxFilterSize`           | $S_{\text{win}}$     | Adaptive window size  | 15      |
| `thres`                   | $\tau_{\text{diff}}$ | Binarization constant | 3       |
| `minSize`                 | $L_{\min}$           | Min. side length (px) | 10      |
| `maxTimesRevisited`       | $R_{\max}$           | Max. pixel revisits   | 4       |
| `maxAttemptsPerCandidate` | $K_{\max}$           | Max. refinement tries | 5       |

---

## 3. Illustrative examples

The design philosophy of ArUco Nano prioritizes ease of integration. As a header-only library, it requires neither complex build systems nor external linking beyond the standard OpenCV modules. To demonstrate its usage, we provide a self-contained, minimal example that covers the two most common tasks in fiducial marker applications: marker detection and 3D pose estimation.

### 3.1. Basic detection and visualization

Listing 1 demonstrates the minimal code required to detect markers in an image. The user simply includes the single header file `aruco_nano.h` and calls the static method `MarkerDetector::detect()`. The function automatically handles the internal pipeline (thresholding, contour extraction, and decoding) and returns a vector of detected markers. Optionally, this function can receive the detection parameters (`aruco_nano::Params`) if the user needs to fine-tune them.

```cpp
#include <opencv2/highgui.hpp>
#include "aruco_nano.h"

int main() {
    // 1. Load the input image
    cv::Mat image = cv::imread("input.jpg");

    // 2. Detect markers
    auto markers = aruco_nano::MarkerDetector::detect(image);

    // 3. Draw results
    for (const auto &m : markers)
        m.draw(image);

    // 4. Save image
    cv::imwrite("output.jpg", image);
    return 0;
}
```

**Listing 1.** Minimal code for marker detection and drawing.

Alternatively, we provide the class `aruco_nano::ArucoDetector`, which mimics the OpenCV API to ease adoption by existing users (see Listing 2).

```cpp
#include <opencv2/highgui.hpp>
#include "aruco_nano.h"

int main() {
    // 1. Load image
    cv::Mat image = cv::imread("/path/to/image.jpg");

    // 2. Create a dictionary
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);

    // 3. Create the detector
    aruco_nano::ArucoDetector detector(dictionary);

    // 4. Vector of marker ids
    std::vector<int> ids;

    // 5. Vector of marker corners
    std::vector<std::vector<cv::Point2f>> corners;

    // 6. Detect markers
    detector.detectMarkers(image, corners, ids);

    // 7. Draw markers to image
    cv::aruco::drawDetectedMarkers(image, corners, ids);

    // 8. Write results
    cv::imwrite("output.jpg", image);
    return 0;
}
```

**Listing 2.** Minimal code for marker detection using OpenCV-like function call.

### 3.2. Pose estimation

For augmented reality or robotics applications, retrieving the marker’s 3D pose is essential. Listing 3 extends Listing 1 to compute the rotation (`rvec`) and translation (`tvec`) vectors relative to the camera. The `Marker` class encapsulates the pose estimation logic through the `estimatePose` method, wrapping OpenCV’s PnP solver for convenience.

```cpp
// assuming markers are already detected
// Camera intrinsic parameters (obtained from calibration)
cv::Mat camMatrix = ...;
cv::Mat distCoeff = ...;

// Physical size of the marker
float markerSize = 0.05f; // 5cm

for (const auto &m : markers) {
    // Compute Rotation and Translation
    auto pose = m.estimatePose(camMatrix, distCoeff, markerSize);

    cv::Mat rvec = pose.first;
    cv::Mat tvec = pose.second;

    // Draw 3D axis
    cv::drawFrameAxes(image, camMatrix, distCoeff, rvec, tvec, markerSize);

    // Use rvec and tvec for AR or navigation tasks...
}
```

**Listing 3.** Estimating 3D pose of detected markers.

Overall, these examples show that the ArUco Nano native API is straightforward to integrate and use, and also provides the familiar high-level API expected by OpenCV users.

---

## 4. Impact

The impact of the proposed software is tightly linked to the ecosystem through which ArUco is most widely deployed: the OpenCV library. Because ArUco is distributed and maintained as part of OpenCV, improvements to its detection pipeline can immediately benefit a broad and diverse user base. OpenCV is widely regarded as the _de facto_ standard library for computer vision in both academia and industry. Its dominance is evidenced not only by the high citation count of the first comprehensive publication explaining it [16], but also by its extensive open-source footprint, with over 85,700 stars and 56,500 forks on GitHub. Other actively maintained fiducial systems exist, notably AprilTag [23] and STag [24]. However, ArUco Nano focuses specifically on improving the ArUco ecosystem, given its widespread adoption through OpenCV. A comprehensive comparison of fiducial marker systems is provided in [15].

The ArUco fiducial marker detection algorithm and its improvements, originally developed by some of the authors of this work [17–19], collectively have over 5100 citations on Google Scholar. Our original standalone implementation (hosted on SourceForge: https://sourceforge.net/projects/aruco/) has accumulated over 277k downloads as of early 2026, demonstrating significant long-term demand for efficient fiducial marker detection. Following the success of the original standalone library, it was integrated into the OpenCV contributions repository (`opencv_contrib`) in 2015, and was migrated to the main OpenCV repository in 2022, where it is maintained under the `objdetect` module.

Since its inclusion in the main OpenCV trunk, direct traffic to our legacy repository has naturally decreased as users migrated to the OpenCV version. The ArUco module is now deployed in countless devices worldwide, from educational robotics kits to industrial automation systems. In this context, the inefficiencies identified in the current implementation (discussed in Section 1) are not merely local optimization issues but systemic bottlenecks affecting a vast user base. ArUco Nano addresses this by offering a drop-in, high-performance alternative that consistently outperforms the current standard implementation in our evaluation, potentially unlocking performance gains for the entire OpenCV ecosystem. As a consequence, we expect this code to be integrated into OpenCV as a new standard for marker detection.

### 4.1. Evaluation dataset

To evaluate our code, we constructed a challenging evaluation dataset comprising 924 high-resolution 16 MP images from the benchmark by Liu et al. [25]. This dataset was chosen for its diversity in textures and scenes. To simulate challenging scenarios, we synthetically inserted several markers into each image at random positions, with random rotations and scales. Crucially, rather than a simple overlay, we modulated the marker intensity based on the local luminance of the original scene. This process blends the markers into the environment, creating complex illumination conditions and shadows that test the robustness of the binarization and identification stages. The dataset contains a total of 9072 ground-truth markers, with an average of 9 markers per image (min: 3, max: 17). In addition, a similar number of non-marker elements resembling valid markers were included to increase scene complexity. While this setup provides controlled and reproducible conditions, it does not capture all real-world challenges, such as motion blur, lens distortion, or specular reflections. Nevertheless, the dataset allows us to stress-test the binarization and identification stages, thereby enabling a valid relative comparison between the tested methods.

![Examples from the evaluation dataset](fig3.png)  
_Fig. 3. Examples from the evaluation dataset. Left: original images with detected markers and their IDs overlaid across cluttered indoor/outdoor scenes. Right: thresholded results used for contour extraction. (See text for details)._

Fig. 3 illustrates some examples of these challenging conditions. The images on the left show the original images with the detected markers overlaid and their IDs, and the images on the right show the thresholded images used for contour extraction.

The dataset contains only markers from the dictionary `DICT_ARUCO_MIP_36h12`. Since the sub-pixel sampling operates on individual code cells independently of the dictionary layout, the approach is expected to generalize to other common dictionaries.

### 4.2. Performance benchmarks

We compared three configurations: standard OpenCV ArUco [26] single-threaded, 20-thread parallel, and ArUco Nano, all on a 13th Gen Intel(R) Core(TM) i7-13700H. To ensure consistent benchmarking and avoid performance variability caused by thermal throttling, we disabled Intel Turbo Boost and set the CPU frequency to 4000 MHz for all tests. For each configuration, experiments were conducted at five resolutions to assess scalability: 16 MP (original), 8 MP, 4 MP, 2 MP, and 1 MP.

### Table 2: Speedup comparison at different resolutions of ArUco Nano and the official OpenCV implementation.

| Method                    | 1 MP            | 2 MP            | 4 MP             | 8 MP              | 16 MP            |
| :------------------------ | :-------------- | :-------------- | :--------------- | :---------------- | :--------------- |
| **ArUco Nano** (1-thread) | 6.68 ms         | 12.40 ms        | 22.93 ms         | 44.98 ms          | 101.67 ms        |
| **OpenCV** (20-threads)   | 21.59 ms (3.2×) | 36.61 ms (2.9×) | 60.08 ms (2.6×)  | 104.33 ms (2.31×) | 221.40 ms (2.2×) |
| **OpenCV** (1-thread)     | 43.46 ms (6.5×) | 74.51 ms (6.0×) | 125.00 ms (5.4×) | 220.25 ms (4.9×)  | 504.27 ms (4.9×) |

The results, summarized in Table 2, demonstrate the significant efficiency of our approach. Each image was processed 15 times with I/O operations excluded from the measurements. The reported runtimes correspond to the minimum time per image, a standard practice to mitigate operating-system scheduling variability. By focusing on the lower bound of the execution time distribution, we ensure that the reported speedups reflect the intrinsic efficiency of the algorithms rather than stochastic interference from background OS processes. While the mean runtimes were slightly higher due to occasional system interrupts, the relative performance gap between ArUco Nano and the OpenCV baseline remained consistent across all statistical measures. A fixed random seed was used for the stochastic corner refinement, ensuring fully deterministic and reproducible results across runs.

At the demanding 16 MP resolution, ArUco Nano is 4.9× faster than the single-threaded OpenCV baseline. This performance gap widens at lower resolutions; for instance, at 1 MP, our method achieves a 6.5× speedup. Notably, our single-threaded implementation is more than 2× faster than the multi-threaded OpenCV version, even when the latter can use up to 20 threads. The improvement is especially significant at 1 MP, where our method obtains a 3.2× speedup.

While the stochastic refinement used during marker identification may increase latency when an image produces many false candidates, its effect on the total runtime is limited in practice because the overall processing time is dominated by adaptive thresholding and contour extraction, whereas marker identification accounts for a smaller fraction of the pipeline. Moreover, the proposed Visited-Aware contour tracing reduces the number of candidates that reach this stage. Consequently, the additional cost of stochastic retries becomes noticeable only when an exceptionally large number of false candidates reach the identification stage. Even under these conditions, ArUco Nano remains faster than the compared OpenCV implementation.

### 4.3. Detection reliability

Efficiency improvements must not compromise robustness. We focused our accuracy analysis on the native 16 MP resolution. Lower-resolution images were excluded from the accuracy metrics because downsampling naturally renders the smallest markers undetectable (below the minimum pixel size required for decoding), making comparisons with the 16 MP ground truth misleading.

### Table 3: Detection accuracy metrics at native 16 MP resolution.

| Method         | TP   | FP  | FN   | P     | R     | F1 score |
| :------------- | :--- | :-- | :--- | :---- | :---- | :------- |
| **OpenCV**     | 5245 | 0   | 3827 | 1.000 | 0.578 | 0.733    |
| **ArUco Nano** | 5980 | 0   | 3092 | 1.000 | 0.659 | 0.795    |

Table 3 shows the True Positives (TP), False Positives (FP), False Negatives (FN), Precision (P), Recall (R), and F1 score of both methods on the dataset. Our method achieves a superior F1 score (0.795) on the 16 MP dataset, primarily due to the stochastic corner refinement (Section 2.2.3), which successfully decodes small-scale markers under challenging illumination conditions that would otherwise be missed. Notably, no false positives were detected, demonstrating that our sub-pixel approach improves recall without compromising reliability.

A detection is counted as a false positive when the system reports a marker ID at a location where no ground-truth marker with that ID exists. The zero false-positive rate is consistent with the dictionary design: `DICT_ARUCO_MIP_36h12` features a large inter-marker Hamming distance, making spurious ID assignments highly unlikely even in the presence of non-marker patterns in the scene.

In conclusion, our tests show that ArUco Nano consistently outperforms the current OpenCV implementation across all tested scenarios.

---

## 5. Conclusions

In this work, we introduce ArUco Nano, a highly optimized, header-only library for fiducial marker detection. By replacing the standard contour extraction with a novel Visited-Aware algorithm and implementing direct sub-pixel code sampling, we effectively eliminate the computational bottlenecks inherent in the standard OpenCV implementation.

Our extensive evaluation on high-resolution imagery up to 16 MP demonstrates that ArUco Nano achieves up to a 6.5× speedup over the single-threaded baseline and outperforms the multi-threaded implementation by more than a factor of two. Furthermore, this efficiency gain does not compromise reliability: our method achieved a higher F1 score (0.795) and detected more valid markers under challenging illumination conditions than the standard approach. These results confirm that ArUco Nano offers a superior performance-to-cost ratio, enabling robust, high-speed tracking on low-power devices with no external dependencies.

Like all contour-based fiducial detectors, ArUco Nano requires markers to occupy a minimum number of pixels for successful decoding, and performance may degrade under extreme motion blur or severe occlusion. These challenges, which are shared by the standard OpenCV implementation, remain open directions for future work.

---

## CRediT authorship contribution statement

- **Sergio Garrido-Jurado:** Writing – review & editing, Writing – original draft, Validation, Software, Resources, Investigation, Conceptualization.
- **Francisco J. Romero-Ramirez:** Writing – review & editing, Software, Methodology, Conceptualization.
- **Rafael Muñoz-Salinas:** Writing – original draft, Supervision, Software, Project administration, Funding acquisition.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the author(s) used Gemini 3.1 Pro to assist in coding algorithms and writing this paper. After using this tool/service, the author(s) reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This research has been funded by the PID2023-147296NB-I00 project of the Ministry of Science, Innovation, and Universities of Spain.

---

## References

1. Zhao K, Song X, Xu Y, Mao R, Fan S, Zhang J. Autonomous path planning logistics sorting robot experimental teaching platform based on aruco marker. In: 2023 4th International Seminar on Artificial Intelligence, Networking and Information Technology (AINIT); 2023. p. 100–4.
2. Orgeira-Crespo P, Rey G, Pousada PR, Aguado-Agelet F. Transport of light parts for logistics supply in industrial manufacturing plants by means of UAV. In: Springer Nature, Switzerland, Cham; 2024. p. 155–80.
3. Radhakrishnan S, Prasanth A, Sowndarya KD, Elngar AA. Drone-Based application of warehouse logistics – a case study. In: CH. vol. 14, John Wiley & Sons, Ltd; 2025. p. 299–327.
4. El-Sheimy N, Li Y. Indoor navigation: state of the art and future trends. Satell Navig 2021; 2: (1).
5. Muñoz-Salinas R, Medina-Carnicer R. UcoSLAM: simultaneous localization and mapping by fusion of keypoints and squared planar markers. Pattern Recognit 2020; 101: 107193.
6. Tsoukalas A, Tzes A, Khorrami F. Relative pose estimation of unmanned aerial systems. In: 2018 26th Mediterranean conference on control and Automation (MED); 2018. p. 155–60.
7. Chen J, Sun C, Zhang A. Autonomous navigation for adaptive unmanned underwater vehicles using fiducial markers. In: 2021 IEEE international conference on Robotics and Automation (ICRA); 2021. p. 9298–304.
8. Čejka J, Bruno F, Skarlatos D, Liarokapis F. Detecting square markers in underwater environments. Remote Sens 2019; 11: (4) 459.
9. Nahangi M, Heins A, McCabe B, Schoellig A. Automated localization of UAVS in gps-denied indoor construction environments using fiducial markers. In: Teizer J (editor), Proceedings of the 35th international symposium on Automation and Robotics in construction (ISARC), international association for Automation and Robotics in construction (IAARC). Taipei, Taiwan; 2018. p. 88–94.
10. Rigter LS, Rijkmans EC, Inderson A, van den Ende RP, Kerkhof EM, Ketelaars M, van Dieren J, Veenendaal RA, van Triest B, Marijnen CA, et al. Eus-guided fiducial marker placement for radiotherapy in rectal cancer: feasibility of two placement strategies and four fiducial types. Endosc Int Open 2019; 7: (11) E1357–E1364.
11. Iocolano M, Blacksburg S, Carpenter T, Repka M, Carbone S, Demircioglu G, Miccio M, Katz A, Haas J. Prostate fiducial marker placement in patients on anticoagulation: feasibility prior to prostate sbrt. Front Oncol 2020; 10.
12. Cai S, Wang X, Chiang F-K. A case study of augmented reality simulation system application in a Chemistry course. Comput Hum Behav 2014; 37: 31–40.
13. Dash AK, Behera SK, Dogra DP, Roy PP. Designing of marker-based augmented reality learning environment for kids using convolutional neural network architecture. Displays 2018; 55: 46–54, advances in Smart Content-Oriented Display Technology.
14. Jurado D, Jurado JM, Ortega L, Feito FR. GEUINF: real-time visualization of indoor facilities using mixed reality. Sensors 2021; 21: (4).
15. Jurado-Rodriguez D, Muñoz-Salinas R, Garrido-Jurado S, Medina-Carnicer R. Planar fiducial markers: a comparative study. Virtual Real 2023; 27: (3) 1733–49.
16. Bradski G. The opencv library. DR. Dobb’s J Softw Tools 2000; 25: 120–5.
17. Garrido-Jurado S, Muñoz Salinas R, Madrid-Cuevas FJ, Marín-Jiménez MJ. Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognit 2014; 47: (6) 2280–92.
18. Romero-Ramirez FJ, Muñoz-Salinas R, Medina-Carnicer R. Speeded up detection of squared fiducial markers. Image Vis Comput 2018; 76: 38–47.
19. Garrido-Jurado S, Muñoz-Salinas R, Madrid-Cuevas F, Medina-Carnicer R. Generation of fiducial marker dictionaries using mixed integer linear programming. Pattern Recognit 2016; 51: 481–91.
20. Suzuki S, Abe K. Topological structural analysis of digitized binary images by border following. Comput Vis Graph Image Process 1985; 30: (1), 32–46.
21. Gonzalez RC, Woods RE. Digital image processing. 4th ed. Pearson, New York, NY; 2018.
22. Douglas DH, Peucker TK. Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. Cartogr: Int J Geogr Inf Geovisualization 1973; 10: (2) 112–22.
23. Wang J, Olson E. AprilTag 2: efficient and robust fiducial detection. In: Proceedings of the IEEE/RSJ international conference on intelligent robots and systems (IROS); 2016.
24. Benligiray B, Topal C, Akinlar C. STag: a stable fiducial marker system. Image Vis Comput 2019; 89: 158–69.
25. Liu J, Liu D, Yang W, Xia S, Zhang X, Dai Y. A comprehensive benchmark for single image compression artifact reduction. IEEE Trans Image Process 2020; 29: 7845–60.
26. Team O. Open source computer vision library; 2025. https://opencv.org. Accessed: 2026-04-.

## References

1. [https://www.researchgate.net/scientific-contributions/Sergio-Garrido-Jurado-2340973335](https://www.researchgate.net/scientific-contributions/Sergio-Garrido-Jurado-2340973335)
