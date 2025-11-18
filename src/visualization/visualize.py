import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Visualize the class distribution
    sns.countplot(x='fraud', data=transaction_data_cleaned)
    plt.title('Fraud Class Distribution')
    plt.show()

    # Visualize the used pin number distribution
    sns.countplot(x='used_pin_number', data=transaction_data_cleaned)
    plt.title('Used Pin Distribution')
    plt.show()

    # Visualize the repeat retailer distribution
    sns.countplot(x='repeat_retailer', data=transaction_data_cleaned)
    plt.title('Repeat Retailer Distribution')
    plt.show()

    # Visualize the distribution of 'ratio_to_median_purchase_price'
    sns.histplot(transaction_data_cleaned['ratio_to_median_purchase_price'], bins=30)
    plt.title('Ratio to Median Purchase Price Distribution')
    plt.show()

    sns.countplot(data=transaction_data_cleaned, x='used_pin_number', hue='fraud', palette=['green', 'red'])
    plt.title('Transactions with PIN vs Fraudulent Transactions')
    plt.xlabel('Used PIN')
    plt.ylabel('Count')
    plt.legend(title='Fraud', labels=['Non-Fraudulent', 'Fraudulent'])
    plt.show()

    # Create a percentage plot to show the percentage of fraudulent transactions when a PIN was used or not
    df_pin_fraud = transaction_data_cleaned.groupby('used_pin_number')['fraud'].value_counts(normalize=True).unstack() * 100
    
    df_pin_fraud.plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(5, 3))
    plt.title('Percentage of Fraudulent Transactions with PIN Usage')
    plt.xlabel('Used PIN')
    plt.ylabel('Percentage')
    plt.legend(title='Fraud', labels=['Non-Fraudulent', 'Fraudulent'])
    plt.show()

    # Create a count plot to show the number of transactions that were fraudulent vs non-fraudulent when a chip was used or not
    plt.figure(figsize=(5, 3))
    sns.countplot(data=transaction_data_cleaned, x='used_chip', hue='fraud', palette=['green', 'red'])
    plt.title('Transactions with Chip vs Fraudulent Transactions')
    plt.xlabel('Used Chip')
    plt.ylabel('Count')
    plt.legend(title='Fraud', labels=['Non-Fraudulent', 'Fraudulent'])
    plt.show()
    
    # Create a percentage plot to show the percentage of fraudulent transactions when a chip was used or not
    df_chip_fraud = transaction_data_cleaned.groupby('used_chip')['fraud'].value_counts(normalize=True).unstack() * 100
    
    df_chip_fraud.plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(5, 3))
    plt.title('Percentage of Fraudulent Transactions with Chip Usage')
    plt.xlabel('Used Chip')
    plt.ylabel('Percentage')
    plt.legend(title='Fraud', labels=['Non-Fraudulent', 'Fraudulent'])
    plt.show()

    # YOUTRY: Fill in the blanks yourself to show the number of transactions that were fraudulent vs non-fraudulent when it was an online order or not
    
    # Create a count plot
    plt.figure(figsize=(5, 3))
    sns.countplot(data=__________, x=__________, hue=__________, palette=['green', 'red'])
    plt.title('Transactions with Online Order vs Fraudulent Transactions')
    plt.xlabel('Online Order')
    plt.ylabel('Count')
    plt.legend(title='Fraud', labels=['Non-Fraudulent', 'Fraudulent'])
    plt.show()
    
    # Create a percentage plot to show the percentage of fraudulent transactions when it was an online order or not
    df_online_order_fraud = transaction_data_cleaned.groupby('online_order')['fraud'].value_counts(normalize=True).unstack() * 100
    
    df_online_order_fraud.plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(10, 6))
    plt.title('Percentage of Fraudulent Transactions with Online Order')
    plt.xlabel('Online Order')
    plt.ylabel('Percentage')
    plt.legend(title='Fraud', labels=['Non-Fraudulent', 'Fraudulent'])
    plt.show()

    # Create a scatter plot to show the breakdown of fraudulent and non-fraudulent transactions
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=transaction_data_cleaned,
        x='distance_from_home',
        y='ratio_to_median_purchase_price',
        hue='fraud',
        palette={0: 'green', 1: 'red'},
    )
    plt.title('Fraudulent vs Non-Fraudulent Transactions: Distance from Home vs Ratio to Median Purchase Price')
    plt.xlabel('Distance from Home')
    plt.ylabel('Ratio to Median Purchase Price')
    plt.show()

    # YOUTRY: Create a scatter plot to show the breakdown of fraudulent and non-fraudulent transactions for distance_from_last_transaction vs ratio_to_median_purchase_price
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=transaction_data_cleaned,
        x='distance_from_last_transaction',
        y=________,
        hue=_____,
        palette={0: 'green', 1: 'red'},
    )
    plt.title('Fraudulent vs Non-Fraudulent Transactions: Distance from Last Transaction vs Ratio to Median Purchase Price')
    plt.xlabel('Distance from Last Transaction')
    plt.ylabel('Ratio to Median Purchase Price')
    plt.show()

    # Confusion matrix for Never Fraud Model
    conf_matrix_never_fraud = confusion_matrix(y_test, y_pred_never_fraud)
    acc = round(100*accuracy_score(y_test, y_pred_never_fraud),1)
    sns.heatmap(conf_matrix_never_fraud, annot=True, fmt='d', cmap='Reds', cbar=True)
    plt.title(f'Never Fraud Model Performance: {acc}% accuracy')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Confusion matrix for k-NN
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    acc = round(100*accuracy_score(y_test, y_pred_knn),1)
    sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'3-NN Performance: {acc}% accuracy')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Compute metrics for "Never Fraud"
    accuracy_never_fraud = accuracy_score(y_test, y_pred_never_fraud)
    precision_never_fraud = precision_score(y_test, y_pred_never_fraud, zero_division=0)
    recall_never_fraud = recall_score(y_test, y_pred_never_fraud)
    f1_never_fraud = f1_score(y_test, y_pred_never_fraud)
    
    # Compute metrics for k-NN
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    precision_knn = precision_score(y_test, y_pred_knn)
    recall_knn = recall_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)
    
    # Prepare data for visualization
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    knn_scores = [accuracy_knn, precision_knn, recall_knn, f1_knn]
    never_fraud_scores = [accuracy_never_fraud, precision_never_fraud, recall_never_fraud, f1_never_fraud]
    
    comparison_transaction_data = pd.DataFrame({
        'Metric': metrics,
        'k-NN': knn_scores,
        'Never Fraud': never_fraud_scores
    })
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    index = range(len(metrics))
    
    bar1 = ax.bar(index, knn_scores, bar_width, label='k-NN', color='blue')
    bar2 = ax.bar([i + bar_width for i in index], never_fraud_scores, bar_width, label='Never Fraud', color='red')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()



if __name__ == '__main__':
    main()
