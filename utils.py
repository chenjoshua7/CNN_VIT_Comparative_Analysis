import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import splitfolders
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import tensorflow as tf


import plotly.graph_objects as go
from plotly.subplots import make_subplots


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Metrics:
    def __init__(self, title):
        self.results = {}
        self.title = title

    def run(self, y_true, y_pred, method_name, average='macro'):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)

        self.results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def plot(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        data = {metric: [res[metric] * 100 for res in self.results.values()] for metric in metrics}
        models = list(self.results.keys())

        # Create subplots
        fig = make_subplots(rows=2, cols=2, subplot_titles=metrics)

        # Add bar charts
        positions = [(i, j) for i in range(1, 3) for j in range(1, 3)]
        for metric, pos in zip(metrics, positions):
            fig.add_trace(
                go.Bar(x=models, y=data[metric], name=metric,
                    text=[f"{v:.2f}" for v in data[metric]], textposition='auto'),
                row=pos[0], col=pos[1]
            )

        # Update layout
        fig.update_layout(
            title_text=self.title,
            height=800, width=1200, 
            
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)

        fig.show()
        
        
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    plt.show()