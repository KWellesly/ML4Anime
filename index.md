# Anime Recommendation Engine

## Motivation [kevin]
Anime is a form of animated media with origins tied to Japan. A recent Google trend revealed that there are between 10-100M searches for anime related topics every month [1]. This number has only just peaked in recent months [2] as a result of nation-wide quarantine orders and subsequent efforts to find an entertainment medium. Our goal is to apply machine learning to recommend the best anime for a user to watch based on their personal favorites. Recommendation engines can be built using the techniques of either collaborative or content-based filtering. Due to the limitations of our dataset, our implementation involved using content-based filtering with a modified KNN. To enhance the model and provide only the best of recommendations, we used a combination of dense, categorical, and textual features.

## Data

### Dataset Description
We created the dataset for our model by combining two kaggle datasets, "Anime Recommendations Database" and "MyAnimeList Dataset." We were able to do this by joining the two datasets on their common animeID feature, and the result allows us to see both rating and demographic information for 13,631 unique anime. In our complete original dataset, we have 77,911 records with each consisting of 28 features. These features include: 
+ <ins>animeID</ins>: Uniquely identifies each of the 13,631 included animes
+ <ins>name</ins>: Anime title
+ <ins>title_english</ins>: Anime title written in English
+ <ins>title_japanese</ins>: Anime title written in Japanese
+ <ins>title_synonyms</ins>: Array containing known nicknames for the anime
+ <ins>type</ins>: Anime type such as Movie, Music, ONA, OVa, Special, TV, or Unknown
+ <ins>source</ins>: Anime source such as Original, Manga, Book, Game, Music, etc. (16 unique)

<p align='center'>
<img src="/ML4Anime/graphs/Type Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="/ML4Anime/graphs/Source Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;"></p>
</p>

<p align='center'>Figure __: Anime Count Comparisons by Type and Source</p>

+ <ins>producers</ins>: Producer of the anime (1073 unique)
+ <ins>genre</ins>: Anime genre such as Action, Sci-Fi, Fantasy, etc. (40 unique)
+ <ins>studio</ins>: The studio creating the anime (47 unique)
+ <ins>episodes</ins>: The number of episodes in the anime (range from 1 to 3057)
+ <ins>status</ins>: Status of "Currently Airing" or "Finished Airing"
+ <ins>airing</ins>: TRUE if Status is "Currently Airing", FALSE otherwise

<p align='center'>
<img src="/ML4Anime/graphs/Genre Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="/ML4Anime/graphs/Airing Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;"></p>
</p>

<p align='center'>
  Figure __: Anime Count Comparison by Genre and Airing Status
</p>

+ <ins>start_date</ins>: Date that the anime started airing (ranges from 1/1/1917 to 2/3/2019)
+ <ins>end_date</ins>: Date that the anime stopped airing (ranges from 2/2/1962 to 9/2/2019)
+ <ins>duration</ins>: Episode length such as 24 min, 1 hr 55 min, etc. (ranges from 7 sec to 3 hr 51 min)
+ <ins>rating</ins>: Audience rating such as None, G, PG, PG-13, R 17+, or R+
+ <ins>score</ins>: Average rating for the anime (ranges from 1 to 10)

<p align='center'>
<img src="/ML4Anime/graphs/Rating Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="/ML4Anime/graphs/Score Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;"></p>
</p>

<p align='center'>Figure __: Anime Count Comparisons by Rating and Score</p>

+ <ins>scored_by</ins>: Number of people who scored the anime (ranges from 0 to 1107955)
+ <ins>rank</ins>: Rank of the anime (ranges from 1 to 13838)
+ <ins>popularity</ins>: Popularity rank according to MyAnimeList.net (ranges from 1 to 15474)
+ <ins>members</ins>: Number of community members in the anime's group (ranges from 6 to 1610561)
+ <ins>favorites</ins>: Number of times the anime has been added to a person's favorites (ranges from 0 to 120331)
+ <ins>synopsis</ins>: Paragraph description of the anime storyline
+ <ins>background</ins>: Paragraph description of the history behind the anime's creation
+ <ins>premiered</ins>: Season that the anime premiered (ranges from Spring 1961 to Winter 2019)
+ <ins>broadcast</ins>: Scheduled broadcast time each week for the anime
+ <ins>related</ins>: Dictionary recording of any known related anime series

### Pre-processing [techniques we used, cleaning text, one-hot encoding, normalizing, graphs, correlation matrix, word embeddings, talk about correlations, etc] - [sanders, stella, savannah]

Before we were able to use the data, we first had to clean it by removing the unnecessary columns and replacing NA values with 0s. Following this, we also one-hot encoded all of the categorical data columns (i.e. genre, studio, source, producers, rating, type). One-hot encoding not only reduced the number of rows in our dataset by ensuring that each anime only occupied one row, but also prepared the dataset for constructing the vectors during the data modelling phase. 

In addition to the categorical data columns, our dataset conveniently held a wealth of information for us in the form of a textual synopsis for each anime. To utilize of this, we used a pretrained word2vec model by Google that was trained on the Google News corpus (over 300 billion words) to output 300-dimensional word vectors. The idea was to use the word embeddings to capture the semantics of the summary in an attempt to use these features to find other anime with similar summaries in semantics. In order to ensure that the input to the model was standardized, the synopsis for each anime was pre-processed to ensure that they were properly formatted and consisted of only words of interest. We removed all punctuations and capitalization, as well as common words such as “a”, “an”, and “in” using the list of default stopwords used by MySQL’s MyISAM search indexes. This significantly reduced the amount of words we were working with as the size of our word bank decreased from 34354 to 21259, and the maximum length of the synopses decreased from 540 to 290. We then computed a 1x300 **synopsis summary vector** for each anime by plugging in every word of the synopsis into the word2vec model and averaging all of the vectors. Note, fictional words specific to an anime (such as "Geass" or names like "Lelouch") may not generate a resulting word embedding, in which case the word is simply ignored in the final calculation of the synopsis summary vector.

<p align='center'>
  <img src="/ML4Anime/graphs/synopsis_summary_vector.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: Synopsis summary vector
</p>

Ultimately, each anime had a corresponding feature vector of shape 1x414. To better understand our feature set and intrinsic relationships amongst features, the following correlation matrices (performed on subsets of features for visibility) were generated:
<p align='center'>
  <img src="/ML4Anime/graphs/stats_genre_corr_matrix.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: Correlation matrix for stats and genre features
</p>

The above *stats* correlation matrix shows many expected behaviors. For example: a very strong negative correlation between score and ranking, and a very strong positive correlation between members and number of favorites. Likewise, there are relatively strong positive correlations between the genres of "Ecchi" and "Harem", and "Fantasy" and "Magic". Particularly interesting was the fact that anime with the genre "Kids" had a much higher chance of being popular while anime labelled as "Romance" were more likely to be less popular. 


<p align='center'>
  <img src="/ML4Anime/graphs/stats_producecr_corr_matrix.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: Correlation matrix for stats and producer features
</p>
The above correlation matrix shows the correlation matrix for the subset of our features containing information on the producer. While there were many producers to consider, the more notable ones: Aniplex, a flagship animation company owned by Sony, and Dentsu, Japan's largest advertising company, had positive correlations with respect to their scores, number of favorites, and number of members. 

### PCA [stella + kevin]

Due to the fact that our feature space was so large (primarily as a result of using textual features), we attempted to reduce the feature space by using PCA. By graphing the summed captured variance of each component, we deduced that using 300 components out of the total 412 was suitable for our needs as it covered 98% of the variance of our feature set. This PCA'ed version of our feature set was then used in our KNN model to find the best anime recommendations. 

<p align='center'>
  <img src="/ML4Anime/graphs/PCA_captured_var.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: Captured variance of 300 components was 98%
</p>



In an attempt to better visualize the feature space, and the relative space and groupings of anime, we used PCA to convert down to 2D space. It is important to note that using 2 features only captures 12.2% of the total variance in our feature set, and thus the feature space visualization is not optimal but merely serves as a visualization to gain a better understanding of the dataset. 
<p align='center'>
  <img src="/ML4Anime/graphs/PCA-2D.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: PCA of feature space into 2D space
</p>

### DBSCAN [kevin]

The PCA graph in 2 dimensional space showed clearly distinct clusters of anime which made us wonder exactly how these clusters formed and what type of anime were represented in each cluster. To tackle this problem, we converted our feature space to 300 dimensions (same feature space as our input to KNN), and performed DBSCAN, an unsupervised clustering algorithm. In order to properly use DBSCAN, we tuned the *minpts* parameter by hand such that not all the points were located in one cluster nor were there an exceptionally large number of noise points. Note, we could not use the heuristic of *minpts* <= D+1, because D would have been set to ~301 or ~13% of our entire dataset. We set *minpts*=3. *Epsilon* was tuned by graphing and sorting the distances of the 10th nearest neighbor of each point in 300 dimensional space. The “elbow method” was used to set *epsilon* to 30.

<p align='center'>
  <img src="/ML4Anime/graphs/DBSCAN_elbow_method.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: Elbow method to tune the epsilon parameter for DBSCAN
</p>

The resulting DBSCAN consisted of 4 clusters and 97 noise points.
<p align='center'>
  <img src="/ML4Anime/graphs/DBSCAN.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: DBSCAN on PCA of feature space
</p>

Below is a deeper dive into a subset of specific anime within each cluster:

<p align='center'>
  <table>
    <thead>
      <tr>
        <th>Cluster 1</th>
        <th>Cluster 2</th>
        <th>Cluster 3</th>
        <th>Cluster 4</th>
        <th>Cluster Outlier</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><img src="/ML4Anime/graphs/cluster1_topk.jpg" width="500"/></td>
        <td><img src="/ML4Anime/graphs/cluster2_topk.jpg" width="500"/></td>
        <td><img src="/ML4Anime/graphs/cluster3_topk.jpg" width="500"/></td>
        <td><img src="/ML4Anime/graphs/cluster4_topk.jpg" width="500"/></td>
        <td><img src="/ML4Anime/graphs/cluster_outlier_topk.jpg" width="500"/></td>
      </tr>
    </tbody>
  </table>
</p>


## Modelling & Results
### Modelling [average of the vector representation of each anime, what distance metric was used, etc]
The KNN algorithm seeks to find the k most similar anime to the current anime. However, often times it is very difficult for users to be able to capture the full breadth of their anime preferences in a single anime. In our modified KNN algorithm, we allow users to input an arbitrary amount of anime that they like in an attempt to better understand and recommend anime catered to their preference. Assume a user inputs *n* different anime that they enjoyed. To model this, we average out the *n* feature vectors of each of those anime and compute KNN on this new vector that ideally captures the essence of each of their preferred animes.
<p align='center'>
  <img src="/ML4Anime/graphs/KNN_input_vector.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: KNN input vector
</p>
<p align='center'>
  <img src="/ML4Anime/graphs/KNN_input.jpg" width="500"/>
</p>
<p align='center'>
  Figure __: Graphical representation of KNN input vector
</p>



### Results [show results of KNN before normalizing/PCA, then after KNN on normalized or PCA'd dataset, show examples of results, no way to validate results] - [linsey]


## Conclusion [Stella]



### References [all of us]

1. Ellis, Theo J. "How the Anime Industry Has Grown Since 2004, According to Google Trends." _Anime Motivation_, animemotivation.com, 23 June 2018, https://animemotivation.com/anime-industry-growth-2004-to-2018/.      

2. Ellis, Theo J. "Why The Coronavirus Has Made Anime More Popular Than Ever." _Anime Motivation_, animemotivation.com, 24 March 2020, https://animemotivation.com/coronavirus-has-made-anime-more-popular/.
