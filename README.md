# CSSM530Work
Dear Professor and Dear Friends!

Welcome to my Github page, where I put the relavant data and the other information concerning my research project for this semester's text-analysis class (CSSM530).

At the .XLSX file, you can find my raw data, which reflects 5,119 Turkish MFA announcements made during the period from September 2012 to April 2025. For a better navigation, I have also implemented their titles, dates, contents, and keywords that I derived through different methods of text analysis respectivelly.

For my research's data collection and classification, I have utilized Python3 codes, which I have obtained and recalibrated through ChatGPT (4o specifically, for a better coding experience as the GPT underlines).

For web scrapping (data collection), I have utilized the below code (as far as I remember, please kindly let me know if there seems any major problems on the code, as there were a lot of different codes I have tried)

---

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

# Configure headless Chrome
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Base URL and entry page
base_url = "https://www.mfa.gov.tr"
url = "https://www.mfa.gov.tr/sub.tr.mfa?3fc6582e-a37b-40d1-847a-6914dc12fb60"
driver.get(url)
time.sleep(3)

# Collect data
all_data = []

def collect_page_data():
    time.sleep(2)
    links = driver.find_elements(By.CSS_SELECTOR, ".sub_lstitm a")
    for link in links:
        title = link.text.strip()
        href = link.get_attribute("href")
        full_link = href if href.startswith("http") else base_url + href
        try:
            date = title.split(",")[1].strip().split(" ")[0]
        except:
            date = "Unknown"
        all_data.append({
            "Date": date,
            "Title": title,
            "Link": full_link
        })

# Start pagination loop
current_page = 1
while True:
    collect_page_data()
    try:
        # Look for the next page by current page + 1
        next_page = driver.find_element(By.XPATH, f"//a[contains(@href, 'Page${current_page + 1}')]")
        driver.execute_script("arguments[0].click();", next_page)
        current_page += 1
        time.sleep(2)
    except:
        break

driver.quit()

# Save to Excel
df = pd.DataFrame(all_data)
output_path = os.path.expanduser("~/Downloads/mfa_announcements_full.xlsx")
df.to_excel(output_path, index=False)
print(f"Data saved to {output_path}")
---



On the other hand, for trend analysis (keywords etc), I have utilized the following code:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import numpy as np
from pathlib import Path

# Load the Excel file
file_path = Path.home() / "Downloads" / "mfa_announcements_with_content.xlsx"
df = pd.read_excel(file_path)

# Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ ]+', '', text)
    return text

df['clean_content'] = df['CONTENT'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['clean_content'])

# KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Extract top terms per cluster
feature_names = vectorizer.get_feature_names_out()
def get_top_terms_per_cluster(tfidf_matrix, labels, top_n=10):
    top_terms = {}
    for cluster_num in np.unique(labels):
        cluster_indices = np.where(labels == cluster_num)[0]
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
        top_indices = np.argsort(cluster_tfidf.A.flatten())[::-1][:top_n]
        top_terms[cluster_num] = [feature_names[i] for i in top_indices]
    return top_terms

top_terms = get_top_terms_per_cluster(tfidf_matrix, df['cluster'].values)
top_terms_df = pd.DataFrame.from_dict(top_terms, orient='index', columns=[f"Keyword {i+1}" for i in range(10)])
top_terms_df.to_csv(Path.home() / "Downloads" / "top_keywords_per_cluster.csv", index_label="Cluster")

# Handle Turkish month names
turkish_to_english = {
    "Ocak": "January", "Şubat": "February", "Mart": "March", "Nisan": "April",
    "Mayıs": "May", "Haziran": "June", "Temmuz": "July", "Ağustos": "August",
    "Eylül": "September", "Ekim": "October", "Kasım": "November", "Aralık": "December"
}

def convert_turkish_date(date_str):
    if isinstance(date_str, str):
        for tr, en in turkish_to_english.items():
            date_str = date_str.replace(tr, en)
    return date_str

df['Date'] = df['Date'].apply(convert_turkish_date)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# Grouping by month and cluster
df['year_month'] = df['Date'].dt.to_period('M')
trend_data = df.groupby(['year_month', 'cluster']).size().unstack(fill_value=0)

# Save trend data
trend_data.to_csv(Path.home() / "Downloads" / "cluster_trends_over_time.csv")

# Check if we have any data to plot
if trend_data.shape[0] > 0 and trend_data.shape[1] > 0:
    plt.figure(figsize=(12, 6))
    trend_data.plot(marker='o')
    plt.title("Trends of Turkish Foreign Policy Clusters Over Time")
    plt.xlabel("Date (Year-Month)")
    plt.ylabel("Number of Announcements")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path.home() / "Downloads" / "cluster_trend_plot.png")
    plt.show()
else:
    print("No valid date or cluster data available for plotting.")
----

I am looking forward for your kind and constructive feedback, criticisms!
Please kindly note that this has been my first ever big-data text-analysis project, and I am sure there have been a considerable mistakes and missing parts. On the other hand, I haven't had time to better integrate LLM codes that I have operationalized (as there seems a necessity to further delve to better recalibrate them), therefore, I will only be able to share them in my article and here after finalizing my work on them.

Thank you very much in advance!

Kindest and Best Regards,
Aybars Arda :)
