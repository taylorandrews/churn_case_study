from eda import load_and_clean_data
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def plot_distance_vs_churn(df):
    y = df.pop('churn').values
    X = df['avg_dist'].values
    x_plot = np.linspace(X.min(), X.max(), 1000)
    kde0 = gaussian_kde(X[y==0])
    kde1 = gaussian_kde(X[y==1])
    y_0 = kde0(x_plot)
    y_1 = kde1(x_plot)
    plt.semilogx(x_plot, y_0, label='not churn')
    plt.semilogx(x_plot, y_1, label='churn')
    plt.xlabel('Average distance in miles')
    plt.ylabel('pdf')
    plt.title('Churn vs Average Ride Distance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/churn_vs_distance.png')
    # plt.show()

def plot_churn_vs_avg_rating_by_driver(df):
    y = df.pop('churn').values
    X = df['avg_rating_by_driver'].values
    x_plot = np.linspace(X.min(), X.max(), 1000)
    kde0 = gaussian_kde(X[y==0])
    kde1 = gaussian_kde(X[y==1])
    y_0 = kde0(x_plot)
    y_1 = kde1(x_plot)
    plt.plot(x_plot, y_0, label='not churn')
    plt.plot(x_plot, y_1, label='churn')
    plt.xlabel('Average Rating By Driver')
    plt.ylabel('pdf')
    plt.title('Churn vs Average Rating By Driver')
    plt.legend()
    plt.savefig('plots/churn_vs_by_driver_rating.png')
    # plt.show()

def plot_churn_vs_avg_rating_of_driver(df):
    y = df.pop('churn').values
    X = df['avg_rating_of_driver'].values
    x_plot = np.linspace(X.min(), X.max(), 1000)
    kde0 = gaussian_kde(X[y==0])
    kde1 = gaussian_kde(X[y==1])
    y_0 = kde0(x_plot)
    y_1 = kde1(x_plot)
    plt.plot(x_plot, y_0, label='not churn')
    plt.plot(x_plot, y_1, label='churn')
    plt.xlabel('Average Rating of Driver')
    plt.ylabel('pdf')
    plt.title('Churn vs Average Rating Of Driver')
    plt.legend()
    plt.savefig('plots/churn_vs_of_driver_rating.png')
    # plt.show()

def plot_churn_vs_surge_percentage(df):
    y = df.pop('churn').values
    X = df['surge_pct'].values
    x_plot = np.linspace(X.min(), X.max(), 1000)
    kde0 = gaussian_kde(X[y==0])
    kde1 = gaussian_kde(X[y==1])
    y_0 = kde0(x_plot)
    y_1 = kde1(x_plot)
    plt.plot(x_plot, y_0, label='not churn')
    plt.plot(x_plot, y_1, label='churn')
    plt.xlabel('Surge Percentage')
    plt.ylabel('pdf')
    plt.title('Churn vs Surge Percentage')
    plt.legend()
    plt.savefig('plots/churn_vs_surge_percentage.png')
    # plt.show()

def plot_churn_vs_average_surge(df):
    y = df.pop('churn').values
    X = df['avg_surge'].values
    x_plot = np.linspace(X.min(), X.max(), 1000)
    kde0 = gaussian_kde(X[y==0])
    kde1 = gaussian_kde(X[y==1])
    y_0 = kde0(x_plot)
    y_1 = kde1(x_plot)
    plt.semilogx(x_plot, y_0, label='not churn')
    plt.semilogx(x_plot, y_1, label='churn')
    plt.xlabel('Average Surge')
    plt.ylabel('pdf')
    plt.title('Churn vs Average Surge')
    plt.legend()
    plt.savefig('plots/churn_vs_average_surge.png')
    # plt.show()

def plot_churn_vs_city(df):
    # y = df.pop('churn').values
    # X = df['city'].values
    icity_churn = dfc['city'][(dfc['city']=='icity')&(dfc['churn']==1)].count()
    android_churn = dfc['city'][(dfc['city']=='Android')&(dfc['churn']==1)].count()
    icity_not_churn = dfc['city'][(dfc['city']=='icity')&(dfc['churn']==0)].count()
    android_not_churn = dfc['city'][(dfc['city']=='Android')&(dfc['churn']==0)].count()
    x = np.array([0,1])
    ax = plt.subplot(111)
    ax.bar(x-0.2, [android_churn, icity_churn], width=0.2, color='blue', label='churn')
    ax.bar(x, [android_not_churn, icity_not_churn], width=0.2, color='red', label='not churn')
    ax.legend(loc=0)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Android'
    labels[1] = 'icity'
    ax.set_xticklabels(labels)
    ax.set_xticks([0, 1])
    plt.savefig('plots/churn_vs_city.png')
    plt.show()

def plot_churn_bar(df, column_name, column_values):
    churn_values = []
    not_churn_values = []
    for val in column_values:
        churn_values.append(dfc[column_name][(dfc[column_name]==val)&(dfc['churn']==1)].count())
        not_churn_values.append(dfc[column_name][(dfc[column_name]==val)&(dfc['churn']==0)].count())
    x = np.array(range(len(column_values)))
    ax = plt.subplot(111)
    ax.bar(x-0.2, churn_values, width=0.2, color='blue', label='churn')
    ax.bar(x, not_churn_values, width=0.2, color='red', label='not churn')
    ax.legend(loc=0)
    ax.set_xticklabels(column_values)
    ax.set_xticks(range(len(column_values)))
    ax.set_title('Churn vs {}'.format(column_name))
    plt.savefig('plots/churn_vs_{}.png'.format(column_name))
    plt.show()

if __name__ == "__main__":
    plt.style.use('fivethirtyeight')
    dfn, dfc = load_and_clean_data()
    # plot_distance_vs_churn(dfc)
    # plot_churn_vs_avg_rating_by_driver(dfn)
    # plot_churn_vs_avg_rating_of_driver(dfn)
    # plot_churn_vs_surge_percentage(dfn)
    # plot_churn_vs_average_surge(dfn)
    # plot_churn_bar(dfc, 'phone', ['iPhone', 'Android'])
    # plot_churn_bar(dfc, 'city', ['Astapor', 'Winterfell', 'King\'s Landing'])
    # plot_churn_bar(dfc, 'luxury_car_user', [True, False])

    yn = dfn.pop('churn').values
    Xn = dfn.values

    yc = dfc.pop('churn').values
    Xc = dfc.values

    X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(Xn, yn)
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(Xc, yc)
