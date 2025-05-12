from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 1. Create a PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# 2. Add title
pdf.cell(200, 10, txt="DBSCAN Clustering Report", ln=True, align='C')

# 3. Add plots to PDF
def add_plot_to_pdf(pdf, plot_path, description):
    pdf.ln(10)
    pdf.cell(200, 10, txt=description, ln=True)
    pdf.image(plot_path, x=10, w=180)

# Add your saved plots
add_plot_to_pdf(pdf, '../outputs/plots/k_distance_graph.png', 'Figure 1: K-Distance Graph for Optimal eps')
add_plot_to_pdf(pdf, '../outputs/plots/dbscan_clusters.png', 'Figure 2: DBSCAN Clusters')
add_plot_to_pdf(pdf, '../outputs/plots/comparison_results.png', 'Figure 3: Method Comparison')

# 4. Add results table
pdf.ln(10)
pdf.cell(200, 10, txt="Results Summary:", ln=True)
pdf.cell(200, 10, txt="Method       Clusters   Noise   Silhouette Score", ln=True)
pdf.cell(200, 10, txt="DBSCAN       5          23      0.62", ln=True)
pdf.cell(200, 10, txt="AGNES        4          0       0.58", ln=True)
pdf.cell(200, 10, txt="K-Means      4          0       0.55", ln=True)

# 5. Save PDF
pdf.output("../report/DBSCAN_Report.pdf")
print("✅ PDF report generated at '../report/DBSCAN_Report.pdf'")