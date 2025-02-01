# FW-S3KIFCM
FW-S3KIFCM: Feature Weighted Safe-Semi-Supervised Kernel-Based Intuitionistic Fuzzy C-Means Clustering Method

Semi-Supervised Clustering (SSC) methods have emerged as a notable research area in machine learning, by integrating prior knowledge into clustering. However, these methods encounter some fundamental issues. The imbalance between labeled and unlabeled data complicates the uncertainty management. This issue is frequently related to numerous real-world problems. On the other hand, existing SSC techniques presume uniform significance for all attributes, disregarding potential variations in feature importance. This presumption hinders the creation of optimal clusters. Furthermore, all existing approaches employ the Euclidean distance metric, which is susceptible to noise and outliers. This paper proposes a robust safe-semi-supervised clustering algorithm, entitled FW-S3KIFCM, to mitigate these shortcomings. For the first time, this approach combines two concepts of Intuitionistic Fuzzy C-Means (IFCM) clustering and Safe-Semi-Supervised Fuzzy C-Means (S3FCM) clustering to address the uncertainty problem in unlabeled data. In addition, it uses a kernel function as a distance metric to tackle noise and outliers. Moreover, incorporating a feature weighting scheme in the objective function highlights the importance of significant features in creating optimal clusters. Experiments on various benchmark datasets demonstrate the method’s superior performance compared to state-of-the-art approaches. 
# Overview of the FW-S3PFCM:
![Untitled](https://github.com/user-attachments/assets/7a1d927d-1242-43f4-b05f-296233e6e483)



# Case study: Brain MRI segmentation 
![image](https://github.com/user-attachments/assets/bf34dd18-5769-4ed4-af37-203ae124f537)

# Comment:
The repository file includes the MATLAB implementation of the FW-S3PFCM algorithm.

Comments are written for all steps of the algorithm for better understanding the code. Also, a demo is implemented for ease of running, which runs by importing the data and other necessary algorithm parameters.

To evaluate the proposed method, the UCI benchmark datasets (https://archive.ics.uci.edu/datasets) and brain MRI segmentation dataset (https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) have been used. 




