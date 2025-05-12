# DBSCAN Clustering Project: Comparison with AGNES and K-Means

![DBSCAN Clusters](outputs/plots/dbscan_clusters.png)  
*Example output from DBSCAN clustering (PCA-reduced 2D projection)*

## 📌 Project Overview
This project compares **DBSCAN** with hierarchical (**AGNES**) and partitioning (**K-Means**) clustering methods on a business dataset. Key steps include:
- Data preprocessing and exploratory analysis (EDA)
- Hyperparameter tuning for DBSCAN (`eps` and `min_samples`)
- Performance evaluation using silhouette scores
- Detailed comparison of all three methods

## 📂 Repository Structure
```plaintext
DBSCAN_Project/
├── data/                   # Input dataset
│   └── BusinessDS.xlsx     # Raw dataset (Excel format)
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 1_EDA_Preprocessing.ipynb
│   ├── 2_DBSCAN_Clustering.ipynb
│   └── 3_Comparison_Analysis.ipynb
├── scripts/                # Python scripts (alternative to notebooks)
│   ├── preprocessing.py
│   ├── dbscan.py
│   └── compare_methods.py
├── outputs/                # Generated results
│   ├── plots/              # Visualizations
│   │   ├── k_distance_graph.png
│   │   ├── dbscan_clusters.png
│   │   └── silhouette_scores.png
│   └── clusters/           # Cluster labels
│       ├── dbscan_labels.csv
│       ├── agnes_labels.csv
│       └── kmeans_labels.csv
├── report/                 # Final documentation
│   ├── DBSCAN_Report.pdf   # Formal PDF report
│   └── README.md           # Project summary (this file)
└── requirements.txt        # Python dependencies
🛠️ Installation & Setup
Clone the repository:

bash
git clone https://github.com/yourusername/DBSCAN-Project.git
cd DBSCAN-Project
Install dependencies:

bash
pip install -r requirements.txt
Launch Jupyter Lab (for notebook users):

bash
jupyter lab

---
## 📊 Key Results
Method	Clusters	Noise Points	Silhouette Score
DBSCAN	5	23	0.62
AGNES	4	0	0.58
K-Means	4	0	0.55
Conclusion: DBSCAN outperformed AGNES and K-Means due to its ability to handle noise and discover non-spherical clusters.

## 📝 Report
DBSCAN_Report.pdf contains:

Methodology details

Visualizations (K-distance graph, cluster plots)

Full comparison analysis

🔧 Dependencies
Python 3.8+

Libraries: See requirements.txt

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
