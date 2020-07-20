import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


initialDiamonds = pd.DataFrame(sns.load_dataset('diamonds'))
print('\nBase dataset:\n')
print(initialDiamonds)
diamonds = initialDiamonds[['carat', 'cut', 'clarity']]
print('\n\nVARIANT 8\nUsed dataset: diamonds(carat, cut, clarity)\n')
print('\nInitial dataset:\n')
print(diamonds)
diamondsCarat = torch.tensor(diamonds.carat)


class ZscoreCategory:

    def __init__(self, dataframe, numericCol, categoryCol):
        self.carat = dataframe[numericCol]
        self.categoryData = dataframe[categoryCol]
        self.mean = self.carat.mean()
        self.sigma = self.carat.std()

    def getScoreCat(self):
        return (self.carat - self.mean) / self.sigma


class Zscore:

    def __init__(self, columns: torch.tensor):
        self.carat = columns
        self.mean = torch.mean(self.carat)
        self.sigma = torch.std(self.carat)

    def getScore(self):
        return (self.carat - self.mean) / self.sigma

    def getAverageScore(self):
        return torch.mean((self.carat - self.mean) / self.sigma)


def checkOutliers(zscorestoCheck):
    return zscorestoCheck['zscores'].apply(lambda x: 'outline' if abs(x) > 3 else 'inlier')


categoryList = ['cut', 'clarity']
# 'Cut' unique values: ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
# 'Clarity' unique values: ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']


scoreCalcCarat = Zscore(diamondsCarat)

print('\n\nAll carat information :')
print('Mean: {0}\nsd: {1}'.format(scoreCalcCarat.mean, scoreCalcCarat.sigma))
print('Z-scores mean: {0}'.format(scoreCalcCarat.getAverageScore()))
zscores = scoreCalcCarat.getScore()
print('Z-scores: {0}'.format(zscores))

for j in categoryList:

    initialDataframe = pd.DataFrame({'carat': scoreCalcCarat.carat, 'zscores': zscores, j: diamonds[j]})

    sns.scatterplot(x=j, y='carat', hue=checkOutliers(initialDataframe), data=initialDataframe)
    plt.title('All z-scores {0}'.format(j))
    plt.xlabel(j)
    plt.ylabel('carat')
    plt.show()

    categoryObj = ZscoreCategory(diamonds, 'carat', j)
    emptyDataframe = pd.DataFrame(columns=['carat', 'zscores', j])
    sourceDataframe = pd.DataFrame({'carat': categoryObj.carat, j: categoryObj.categoryData})

    for i in diamonds[j].unique().tolist():
        tempDataframe = ZscoreCategory(sourceDataframe.loc[sourceDataframe[j] == i], 'carat', j)
        zscoresCat = pd.DataFrame({'carat': tempDataframe.carat, 'zscores': tempDataframe.getScoreCat(), j: i})
        emptyDataframe = emptyDataframe.append(zscoresCat, ignore_index=True)

    sns.scatterplot(x=j, y='carat', hue=checkOutliers(emptyDataframe), data=emptyDataframe)
    plt.title('Category z-scores {0}'.format(j))
    plt.xlabel(j)
    plt.ylabel('carat')
    plt.show()
