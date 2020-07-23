# NETFLIX-MOVIE-RECOMMENDATION-SYSTEM

[![Makes people smile](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://github.com/iamsivab)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fiamsivab%2FMovie-Recommendation-Netflix)](https://hits.seeyoufarm.com)

# Movie-Recommendation-Netflix

[![Generic badge](https://img.shields.io/badge/Datascience-Beginners-Red.svg?style=for-the-badge)](https://github.com/iamsivab/Movie-Recommendation-Netflix) 
[![Generic badge](https://img.shields.io/badge/LinkedIn-Connect-blue.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/iamsivab/) [![Generic badge](https://img.shields.io/badge/Python-Language-blue.svg?style=for-the-badge)](https://github.com/iamsivab/Movie-Recommendation-Netflix) [![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

### Business Problem 
### Problem Description 
Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while Cinematch is doing pretty well, it can always be made better.

Now there are a lot of interesting alternative approaches to how Cinematch works that netflix haven’t tried. Some are described in the literature, some aren’t. We’re curious whether any of these can beat Cinematch by making better predictions. Because, frankly, if there is a much better approach it could make a big difference to our customers and our business.

> Credits: https://www.netflixprize.com/rules.html

#### The goal of this project is to develop a recomendation system [#DataScience](https://github.com/iamsivab/Movie-Recommendation-Netflix) for Netflix.

[![GitHub repo size](https://img.shields.io/github/repo-size/iamsivab/Movie-Recommendation-Netflix.svg?logo=github&style=social)](https://github.com/iamsivab) [![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/iamsivab/Movie-Recommendation-Netflix.svg?logo=git&style=social)](https://github.com/iamsivab/)[![GitHub top language](https://img.shields.io/github/languages/top/iamsivab/Movie-Recommendation-Netflix.svg?logo=python&style=social)](https://github.com/iamsivab)

#### Few popular hashtags - 
### `#DataScience` `#Netflix` `#Recommendation System`
### `#Ratings` `#Movie PRediction` `#Numpy-Pandas`

### Motivation
Netflix provided a lot of anonymous rating data, and a prediction accuracy bar that is 10% better than what Cinematch can do on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.)

### About the Project
1. Predict the rating that a user would give to a movie that he ahs not yet rated.
2. Minimize the difference between predicted and actual rating (RMSE and MAPE) 

### Steps involved in this project

[![Made with Python](https://forthebadge.com/images/badges/made-with-python.svg)](https://github.com/iamsivab/Movie-Recommendation-Netflix) [![Made with love](https://forthebadge.com/images/badges/built-with-love.svg)](https://www.linkedin.com/in/iamsivab/) [![ForTheBadge built-with-swag](http://ForTheBadge.com/images/badges/built-with-swag.svg)](https://www.linkedin.com/in/iamsivab/)

1. Some form of interpretability.
2. Machine Learning Problem 
3. Data 
### Data Overview 
> Get the data from : https://www.kaggle.com/netflix-inc/netflix-prize-data/data

Data files :

- combined_data_1.txt
- combined_data_2.txt
- combined_data_3.txt
- combined_data_4.txt
- movie_titles.csv
  
The first line of each file [combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt] contains the movie id followed by a colon. Each subsequent line in the file corresponds to a rating from a customer and its date in the following format:

CustomerID,Rating,Date

MovieIDs range from 1 to 17770 sequentially.
CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
Ratings are on a five star (integral) scale from 1 to 5.
Dates have the format YYYY-MM-DD.

``` Python
# Movie by Movie Similarity Matrix
start = datetime.now()
if not os.path.isfile('m_m_sim_sparse.npz'):
    print("It seems you don't have that file. Computing movie_movie similarity...")
    start = datetime.now()
    m_m_sim_sparse = cosine_similarity(X=train_sparse_matrix.T, dense_output=False)
    print("Done..")
    # store this sparse matrix in disk before using it. For future purposes.
    print("Saving it to disk without the need of re-computing it again.. ")
    sparse.save_npz("m_m_sim_sparse.npz", m_m_sim_sparse)
    print("Done..")
else:
    print("It is there, We will get it.")
    m_m_sim_sparse = sparse.load_npz("m_m_sim_sparse.npz")
    print("Done ...")

print("It's a ",m_m_sim_sparse.shape," dimensional matrix")

print(datetime.now() - start)

```

### Mapping the real world problem to a Machine Learning Problem 
### Type of Machine Learning Problem 
- For a given movie and user we need to predict the rating would be given by him/her to the movie. 
- The given problem is a Recommendation problem 
- It can also seen as a Regression problem 
### Performance metric 
- Mean Absolute Percentage Error: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
- Root Mean Square Error: https://en.wikipedia.org/wiki/Root-mean-square_deviation
### Machine Learning Objective and Constraints 
- Minimize RMSE.
- Try to provide some interpretability.


### Libraries Used

![Ipynb](https://img.shields.io/badge/Python-datetime-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-pandas-blue.svg?style=flat&logo=python&logoColor=white)
![Ipynb](https://img.shields.io/badge/Python-numpy-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-matplotlib-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-seaborn-blue.svg?style=flat&logo=python&logoColor=white)
![Ipynb](https://img.shields.io/badge/Python-scipy-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-sklearn-blue.svg?style=flat&logo=python&logoColor=white) 


### Installation

- Install **datetime** using pip command: `from datetime import datetime`
- Install **pandas** using pip command: `import pandas as pd`
- Install **numpy** using pip command: `import numpy as np`
- Install **matplotlib** using pip command: `import matplotlib`
- Install **matplotlib.pyplot** using pip command: `import matplotlib.pyplot as plt`
- Install **seaborn** using pip command: `import seaborn as sns`
- Install **os** using pip command: `import os`
- Install **scipy** using pip command: `from scipy import sparse`
- Install **scipy.sparse** using pip command: `from scipy.sparse import csr_matrix`
- Install **sklearn.decomposition** using pip command: `from sklearn.decomposition import TruncatedSVD`
- Install **sklearn.metrics.pairwise** using pip command: `from sklearn.metrics.pairwise import cosine_similarity`
- Install **random** using pip command: `import random`


### How to run?

[![Ipynb](https://img.shields.io/badge/Ipynb-Movie_recommendation.ipynb-lightgrey.svg?logo=python&style=social)](https://github.com/iamsivab/Movie-Recommendation-Netflix/)

``` svd               1.0726046873826458
knn_bsl_u         1.0726493739667242
knn_bsl_m          1.072758832653683
svdpp             1.0728491944183447
bsl_algo          1.0730330260516174
xgb_knn_bsl_mu    1.0753229281412784
xgb_all_models     1.075480663561971
first_algo        1.0761851474385373
xgb_bsl           1.0763419061709816
xgb_final         1.0763580984894978
xgb_knn_bsl       1.0763602465199797
Name: rmse, dtype: object
```


### Project Reports

[![report](https://img.shields.io/static/v1.svg?label=Project&message=Report&logo=microsoft-word&style=social)](https://github.com/iamsivab/Movie-Recommendation-Netflix/)

- [Download](https://github.com/iamsivab/Movie-Recommendation-Netflix/) for the report.

### Useful Links

- https://www.netflixprize.com/rules.html
- https://www.kaggle.com/netflix-inc/netflix-prize-data
- Netflix blog: https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 (very nice blog)
- surprise library: http://surpriselib.com/ (we use many models from this library)
- surprise library doc: http://surprise.readthedocs.io/en/stable/getting_started.html (we use many models from this library)
- installing surprise: https://github.com/NicolasHug/Surprise#installation
- Research paper: http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf (most of our work was inspired by this paper)
- SVD Decomposition : https://www.youtube.com/watch?v=P5mlg91as1c

[![IPYNB](https://img.shields.io/static/v1.svg?label=IPYNB&message=Movie&color=lightgray&logo=linkedin&style=social&colorA=critical)](https://www.linkedin.com/in/iamsivab/) [![GitHub top language](https://img.shields.io/github/languages/top/iamsivab/Movie-Recommendation-Netflix.svg?logo=php&style=social)](https://github.com/iamsivab/)

[Report](https://github.com/iamsivab/Movie-Recommendation-Netflix) - A Detailed Report on the Analysis


### Contributing

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?logo=github)](https://github.com/iamsivab/Movie-Recommendation-Netflix/pulls) [![GitHub issues](https://img.shields.io/github/issues/iamsivab/Movie-Recommendation-Netflix?logo=github)](https://github.com/iamsivab/Movie-Recommendation-Netflix/issues) ![GitHub pull requests](https://img.shields.io/github/issues-pr/viamsivab/Movie-Recommendation-Netflix?color=blue&logo=github) 
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/iamsivab/Movie-Recommendation-Netflix?logo=github)](https://github.com/iamsivab/Movie-Recommendation-Netflix/)

- Clone [this](https://github.com/iamsivab/Movie-Recommendation-Netflix/) repository: 

```bash
git clone https://github.com/iamsivab/Movie-Recommendation-Netflix.git
```

- Check out any issue from [here](https://github.com/iamsivab/Movie-Recommendation-Netflix/issues).

- Make changes and send [Pull Request](https://github.com/iamsivab/Movie-Recommendation-Netflix/pulls).
 
### Need help?

[![Facebook](https://img.shields.io/static/v1.svg?label=follow&message=@iamsivab&color=9cf&logo=facebook&style=flat&logoColor=white&colorA=informational)](https://www.facebook.com/iamsivab)  [![Instagram](https://img.shields.io/static/v1.svg?label=follow&message=@iamsivab&color=grey&logo=instagram&style=flat&logoColor=white&colorA=critical)](https://www.instagram.com/iamsivab/) [![LinkedIn](https://img.shields.io/static/v1.svg?label=connect&message=@iamsivab&color=success&logo=linkedin&style=flat&logoColor=white&colorA=blue)](https://www.linkedin.com/in/iamsivab/)

:email: Feel free to contact me @ [balasiva001@gmail.com](https://mail.google.com/mail/)

[![GMAIL](https://img.shields.io/static/v1.svg?label=send&message=balasiva001@gmail.com&color=red&logo=gmail&style=social)](https://www.github.com/iamsivab) [![Twitter Follow](https://img.shields.io/twitter/follow/iamsivab?style=social)](https://twitter.com/iamsivab)


### License

MIT &copy; [Sivasubramanian](https://github.com/iamsivab/Movie-Recommendation-Netflix/blob/master/LICENSE)

[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/0)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/0)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/1)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/1)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/2)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/2)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/3)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/3)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/4)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/4)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/5)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/5)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/6)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/6)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/images/7)](https://sourcerer.io/fame/iamsivab/iamsivab/Movie-Recommendation-Netflix/links/7)


[![GitHub license](https://img.shields.io/github/license/iamsivab/Movie-Recommendation-Netflix.svg?style=social&logo=github)](https://github.com/iamsivab/Movie-Recommendation-Netflix/blob/master/LICENSE) 
[![GitHub forks](https://img.shields.io/github/forks/iamsivab/Movie-Recommendation-Netflix.svg?style=social)](https://github.com/iamsivab/Movie-Recommendation-Netflix/network) [![GitHub stars](https://img.shields.io/github/stars/iamsivab/Movie-Recommendation-Netflix.svg?style=social)](https://github.com/iamsivab/Movie-Recommendation-Netflix/stargazers) [![GitHub followers](https://img.shields.io/github/followers/iamsivab.svg?label=Follow&style=social)](https://github.com/iamsivab/)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/iamsivab/ama)


