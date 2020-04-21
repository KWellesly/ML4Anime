# Anime Recommendation Engine

## Motivation
Anime is a form of animated media with origins tied to Japan. A recent Google trend revealed that there are between 10-100M searches for anime related topics every month [1]. This number has only just peaked in recent months as a result of nation-wide quarantine orders and subsequent efforts to find an entertainment medium [2]. Our goal is to apply machine learning to recommend the best anime for a user to watch based on their personal favorites. Recommendation engines can be built using the techniques of either collaborative or content-based filtering. Due to the limitations of our dataset, our implementation involved using content-based filtering with a modified KNN. To enhance the model and provide only the best of recommendations, we used a combination of dense, categorical, and textual features.

## Data

### Dataset Description
We utilized a dataset that we found through a GitHub project called tidy.csv that had been constructed from cleaning a kaggle dataset [3]. In our complete original dataset, we have 77,911 records with each consisting of 28 features. These features include: 

|Name|Description|Type|
|---|---|---|
|animeID|Uniquely identifies each of the 13,631 included animes|dense| 
|name|Anime title|categorical|  
|title_english|Anime title in English|categorical|    
|title_japanese|Anime title in Japanese|categorical|
|title_synonyms|Nicknames or other known names for the anime|categorical|
|type|Anime type such as Movie, Music, ONA, OVA, Special, TV, or Unknown|categorical|
|source|Anime source such as Original, Manga, Book, Game, Music, etc. (16 unique)|categorical|
|producers|Producer of the anime (1,073 unique)|categorical|
|genre|Anime genre such as Action, Sci-Fi, Fantasy, etc. (40 unique)|categorical|
|studio|The studio creating the anime (47 unique)|categorical|
|episodes|The number of episodes in the anime (range from 1 to 3,057)|dense|
|status|Status of "Currently Airing" or "Finished Airing"|categorical|
|airing|TRUE if Status is "Currently Airing", FALSE otherwise|categorical|
|start_date|Date that the anime started airing (ranges from 1/1/1917 to 2/3/2019) formatted as ymd|categorical|
|end_date|Date that the anime stopped airing (ranges from 2/2/1962 to 9/2/2019) formatted as ymd|categorical|
|duration|Episode length such as 24 min, 1 hr 55 min, etc. (ranges from 7 sec to 3 hr 51 min)|categorical|
|rating|Audience rating such as None, G, PG, PG-13, R 17+, or R+|categorical|
|score|Average rating for the anime (ranges from 1 to 10)|dense|
|scored_by|Number of people who scored the anime (ranges from 0 to 1,107,955)|dense|
|rank|Rank of the anime (ranges from 1 to 13,838)|dense|
|popularity|Popularity rank according to MyAnimeList.net (ranges from 1 to 15,474)|dense|
|members|Number of community members in the anime's group (ranges from 6 to 1,610,561)|dense|
|favorites|Number of times the anime has been added to a person's favorites (ranges from 0 to 120,331)|dense|
|synopsis|Paragraph description of the anime storyline|textual|
|background|Paragraph description of the history behind the anime's creation|textual|
|premiered|Season that the anime premiered (ranges from Spring 1961 to Winter 2019)|categorical|
|broadcast|Scheduled broadcast time each week for the anime|categorical|
|related|Any known related anime series|categorical|

<p align='center'>
<img src="/ML4Anime/graphs/Type Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="/ML4Anime/graphs/Source Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;"></p>
</p>

<p align='center'>Figure 1: Anime Count Comparisons by Type and Source</p>

<p align='center'>
<img src="/ML4Anime/graphs/Genre Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="/ML4Anime/graphs/Airing Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;"></p>
</p>

<p align='center'>
  Figure 2: Anime Count Comparison by Genre and Airing Status
</p>

<p align='center'>
<img src="/ML4Anime/graphs/Rating Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="/ML4Anime/graphs/Score Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;"></p>
</p>

<p align='center'>Figure 3: Anime Count Comparisons by Rating and Score</p>

<p align='center'>
<img src="/ML4Anime/graphs/Producer Score Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="/ML4Anime/graphs/Studio Score Chart.PNG" style="float: left; width: 49%; margin-right: 1%; margin-bottom: 0.5em;">
<p style="clear: both;"></p>
</p>

<p align='center'>Figure 4: Average Score of Most Reviewed Producers and Studios</p>

<p align='center'>
  <img src="/ML4Anime/graphs/Premier Decade.PNG" width="500"/>
</p>

<p align='center'>Figure 5: Anime Count by Decade of Premier</p>

### Pre-processing

Before we were able to use the data, we first had to clean it by removing the unnecessary columns and replacing NA values with 0s. Although our dataset had 77,911 rows, many of these rows were duplicated multiple times for a single anime title. For example, the anime Cowboy Bebop was duplicated 17 times, once for each genre, each studio, and/or each producer that worked on the anime. To clean this up, we grouped all the anime together by title, and consolidated the information to remove the duplicated rows - ultimately condensing our dataset from 77,911 rows to 2,856 unique anime. Following this, we also one-hot encoded all of the categorical data columns (i.e. genre, studio, source, producers, rating, type). One-hot encoding not only reduced the number of rows in our dataset by ensuring that each anime only occupied one row, but also prepared the dataset for constructing the vectors during the data modelling phase.

In addition to the categorical data columns, our dataset conveniently held a wealth of information for us in the form of a textual synopsis for each anime. To utilize of this, we used a pretrained word2vec model by Google that was trained on the Google News corpus (over 300 billion words) to output 300-dimensional word vectors. The idea was to use the word embeddings to capture the semantics of the summary in an attempt to use these features to find other anime with similar summaries in semantics. In order to ensure that the input to the model was standardized, the synopsis for each anime was pre-processed to ensure that they were properly formatted and consisted of only words of interest. We removed all punctuations and capitalization, as well as common words such as “a”, “an”, and “in” using the list of default stopwords used by MySQL’s MyISAM search indexes [4]. This significantly reduced the amount of words we were working with as the size of our word bank decreased from 34354 to 21259, and the maximum length of the synopses decreased from 540 to 290. We then computed a 1x300 **synopsis summary vector** for each anime by plugging in every word of the synopsis into the word2vec model and averaging all of the vectors. Note, fictional words specific to an anime (such as "Geass" or names like "Lelouch") may not generate a resulting word embedding, in which case the word is simply ignored in the final calculation of the synopsis summary vector.

<p align='center'>
  <img src="/ML4Anime/graphs/synopsis_summary_vector.jpg" width="500"/>
</p>
<p align='center'>
  Figure 6: Synopsis summary vector
</p>

Ultimately, each anime had a corresponding feature vector of shape 1x414. To better understand our feature set and intrinsic relationships amongst features, the following correlation matrices (performed on subsets of features for visibility) were generated:
<p align='center'>
  <img src="/ML4Anime/graphs/stats_genre_corr_matrix.jpg" width="500"/>
</p>
<p align='center'>
  Figure 7: Correlation matrix for stats and genre features
</p>

The above *stats* correlation matrix shows many expected behaviors. For example: a very strong negative correlation between score and ranking, and a very strong positive correlation between members and number of favorites. Likewise, there are relatively strong positive correlations between the genres of "Ecchi" and "Harem", and "Fantasy" and "Magic". Particularly interesting was the fact that anime with the genre "Kids" had a much higher chance of being popular while anime labelled as "Romance" were more likely to be less popular. 


<p align='center'>
  <img src="/ML4Anime/graphs/stats_producecr_corr_matrix.jpg" width="500"/>
</p>
<p align='center'>
  Figure 8: Correlation matrix for stats and producer features
</p>
The above correlation matrix shows the correlation matrix for the subset of our features containing information on the producer. While there were many producers to consider, the more notable ones: Aniplex, a flagship animation company owned by Sony, and Dentsu, Japan's largest advertising company, had positive correlations with respect to their scores, number of favorites, and number of members. 

### PCA

Due to the fact that our feature space was so large (primarily as a result of using textual features), we attempted to reduce the feature space by using PCA. By graphing the summed captured variance of each component, we deduced that using 300 components out of the total 412 was suitable for our needs as it covered 98% of the variance of our feature set. This PCA'ed version of our feature set was then used in our KNN model to find the best anime recommendations. 

<p align='center'>
  <img src="/ML4Anime/graphs/PCA_captured_var.jpg" width="500"/>
</p>
<p align='center'>
  Figure 9: Captured variance of 300 components was 98%
</p>



In an attempt to better visualize the feature space, and the relative space and groupings of anime, we used PCA to convert down to 2D space. It is important to note that using 2 features only captures 12.2% of the total variance in our feature set, and thus the feature space visualization is not optimal but merely serves as a visualization to gain a better understanding of the dataset. 
<p align='center'>
  <img src="/ML4Anime/graphs/PCA-2D.jpg" width="500"/>
</p>
<p align='center'>
  Figure 10: PCA of feature space into 2D space
</p>

### DBSCAN

The PCA graph in 2 dimensional space showed clearly distinct clusters of anime which made us wonder exactly how these clusters formed and what type of anime were represented in each cluster. To tackle this problem, we converted our feature space to 300 dimensions (same feature space as our input to KNN), and performed DBSCAN, an unsupervised clustering algorithm. In order to properly use DBSCAN, we tuned the *minpts* parameter by hand such that not all the points were located in one cluster nor were there an exceptionally large number of noise points. Note, we could not use the heuristic of minpts <= D+1, because D would have been set to ~301 or ~13% of our entire dataset. We set *minpts*=3. *Epsilon* was tuned by graphing and sorting the distances of the 10th nearest neighbor of each point in 300 dimensional space. The “elbow method” was used to set *epsilon* to 30.

<p align='center'>
  <img src="/ML4Anime/graphs/DBSCAN_elbow_method.jpg" width="500"/>
</p>
<p align='center'>
  Figure 11: Elbow method to tune the epsilon parameter for DBSCAN
</p>

The resulting DBSCAN consisted of 4 clusters and 97 noise points. Below is a representation in 2D space.
<p align='center'>
  <img src="/ML4Anime/graphs/DBSCAN.jpg" width="500"/>
</p>
<p align='center'>
  Figure 12: DBSCAN on PCA of feature space
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
<p align='center'>
  Figure 13: Top-15 anime represented in each cluster
</p>

## Modelling & Results
### Modelling 
The KNN algorithm seeks to find the k most similar anime to the current anime. However, often times it is very difficult for users to be able to capture the full breadth of their anime preferences in a single anime. In our modified KNN algorithm, we allow users to input an arbitrary amount of anime that they like in an attempt to better understand and recommend anime catered to their preference. Assume a user inputs *n* different anime that they enjoyed. To model this, we average out the *n* feature vectors of each of those anime and compute KNN on this new vector that ideally captures the essence of each of their preferred animes.
<p align='center'>
  <img src="/ML4Anime/graphs/KNN_input_vector.jpg" width="500"/>
</p>
<p align='center'>
  Figure 14: KNN input vector
</p>
<p align='center'>
  <img src="/ML4Anime/graphs/KNN_input.jpg" width="500"/>
</p>
<p align='center'>
  Figure 15: Graphical representation of KNN input vector
</p>

There were two distance metrics that we considered for our modelling. The first, and preferred method, was using cosine similarity. Cosine distance is defined as:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?%5Ccos%5Ctheta%20%3D%20%5Cfrac%7B%5Coverrightarrow%7Ba%7D%5Ccdot%20%5Coverrightarrow%7Bb%7D%7D%7B%5Cleft%20%5C%7C%20%5Coverrightarrow%7Ba%7D%20%5Cright%20%5C%7C%5Cleft%20%5C%7C%20%5Coverrightarrow%7Bb%7D%20%5Cright%20%5C%7C%7D"/>
</p>
and measures the angle between our input average feature vector and each of the feature vectors for anime in the dataset. We preferred cosine similarity as a distance measurement due to the way our dataset values were distributed.
To process our data, we one-hot encoded our categorical data values, like genre, studio, and source. These columns were represented in our processed data in 1s and 0s. In comparison, our originally quantitative feature data values, such as episodes, which had values ranging from 1 to 1787, and scored_by, with minimum at 8 and maximum value 1107995, were much greater than our one-hot encoded values, and could possibly skew our KNN results towards the originally quantitative features. With this in mind, we implemented Cosine similarity as a distance measurement because it focuses on the angle between the vectors, and does not consider the respective weights or magnitudes of the vectors.
<p align='center'>
  <img src="/ML4Anime/graphs/anime_df_head.jpg" width="500"/>
</p>
<p align='center'>
  Figure 16: Anime Dataset example data, genre_Action (far right) is an example of one-hot encoding of categorical feature genre
</p>

Our alternative distance metric was using Euclidean distance, measured by:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?d%5Cleft%20%28%20x%2Cy%20%5Cright%20%29%3D%5Csqrt%7B%5Cleft%20%5C%7C%20%5Coverrightarrow%7Ba%7D-%5Coverrightarrow%7Bb%7D%20%5Cright%20%5C%7C%5E%7B2%7D%7D"/>
</p>
Euclidean distance, in contrast to Cosine distance, is similar to measuring the actual distance between the two vectors, and is thus affected by angle and magnitude of the vectors. We implemented Euclidean distance as an alternative distance measurement because we were interested in seeing how the different distance functions would perform comparatively to each other.

For our KNN implementation, we compare the distance values of each feature vector to our input average vector. When considering Euclidean distance, this can be compared directly (ex. d(x1,average) = 7.8 < 12 = d(x2,average)). However, the same does not apply for Cosine similarity. A Cosine similarity value (CosTheta) of 0 actually corresponds to an angle of 90 degrees, while a Cosine similarity of 1 corresponds with 0, so they cannot be compared as is. Specifically, we have to shift our Cosine similarity such that a low Cosine distance value corresponds with a low angle. We chose to implement this by representing Cosine distance as:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?1-%5Ccos%5Ctheta%20%3D%201-%5Cfrac%7B%5Coverrightarrow%7Ba%7D%5Ccdot%20%5Coverrightarrow%7Bb%7D%7D%7B%5Cleft%20%5C%7C%20%5Coverrightarrow%7Ba%7D%20%5Cright%20%5C%7C%5Cleft%20%5C%7C%20%5Coverrightarrow%7Bb%7D%20%5Cright%20%5C%7C%7D"/>
</p>
which then ensures minimum angle, 0 degrees, is represented as 1-Cos(0) and thus a minimum Cosine distance value of 0 as well. In contrast, now for an angle of 90 degrees, Cosine distance = 1-Cos(90) = 1-Cos(-90) = 1, and for an angle of 180 degrees, Cosine distance = 1-Cos(180) = 2, the maximum Cosine distance value.

### Results 
Because we are using an unsupervised learning model, there is no sure-fire way to measure the "accuracy" of our results. However, we came up with several comparitive statistical measurements to analyze our recommendations relative to our inputs.

The first measurement we use is the standardized average distance between the average of the input feature vectors and the feature vector for the anime in question, which we refer to as STD Distance. STD Distance takes its derivation from standard deviation and is calculated by:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?%5Csqrt%7B%20%5Cfrac%7B%5Csum%20%5Cleft%20%28%20x_%7Bi%7D%20-%20u%20%5Cright%20%29%5E%7B2%7D%7D%7Bn%7D%20%7D"/>
</p>
where for all xi in the set of result anime feature vectors, n is the number of recommended animes, and u is the average feature vector generated from the input animes. It essentially represents the average variance of the set of recommended anime features from the input average features.

Another metric that we use to compare our overall recommendations to our input animes is average distance:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?%5Cbar%7Bd%7D%3D%5Cfrac%7B%5Csum%20d_%7Bi%7D%7D%7Bn%7D"/>
</p>
and is simply the mean value of all the distances of our output anime.

For our feature comparisons, we defined two major measurements. The first is Average Absolute Standard Z-score of the feature of the output anime. Standard Z-score of anime i refers to the  standardized difference of the value of the anime feature from the mean input feature value and is defined by:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?z_%7Bi%2Cf%7D%20%3D%20%5Cfrac%7Bx_%7Bf%7D-%5Cmu%20%7D%7B%5Csigma%20%7D"/>
</p>
where xf is the value of feature f in anime feature vector x, mu is the average value for that feature from our input animes, and sigma is the standard deviation of that feature from all our of data values.
From Standard Z-score, we define Average Absolute Standard Z as:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?Z_%7Bf%7D%20%3D%20%5Cfrac%7B%20%5Csum%20%5Cleft%20%7C%20z_%7Bi%2Cf%7D%20%5Cright%20%7C%7D%7B%5Csigma%20%7D"/>
</p>
Our second primary metric for feature comparisons is Average Standard Feature Deviation, derived similarly to standard deviation:
<p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?s%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Csum%20%5Cleft%20%28%20f_%7Bi%7D%20-%20%5Cmu%20%5Cright%20%29%5E%7B2%7D%7D%7Bn%7D%7D"/>
</p>
where fi is the feature value of the output anime, mu is the average value for that feature from our input animes, and n is the number of output anime.



 EXAMPLE 1: From a single anime title: ['Attack on Titan']

|               | Cosine Unaltered                                                                                                                                                           | Cosine Normalized                                                                                                                                                                 | Euclidean Unaltered                                                                                                                                                                    |                                                                                    Euclidean Normalized                                                                                   |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 1.11e-16                                                                                                                                                                   | 2.22e-16                                                                                                                                                                          | 0                                                                                                                                                                                      | 0                                                                                                                                                                                         |
| **Distances**     | - **Sword Art Online**: 4.53e-05<br>- **Dragon Ball Z**: 4.82e-05<br>- **Code Geass: Lelouch R2:** 5.28e-05<br>- **Death Note**: 5.83e-05<br>- **One Punch Man**: 1.59e-04 | - **Attack on Titan S2**: 0.26<br>- **Fullmetal Alchemist: Brotherhood**: 0.36<br>- **Death Note**: 0.38<br>- **Code Geass: Lelouch**: 0.40<br>- **Code Geass: Lelouch R2**: 0.44 | - **Sword Art Online**: 68802.63<br>- **Death Note**: 132434.60<br>- **Fullmetal Alchemist: Brotherhood**: 261364.26<br>- **One Punch Man**: 384929.08<br>- **Tokyo Ghoul**: 459418.36 | - **Attack on Titan S2**: 17.51<br>- **Code Geass: Lelouch**: 21.16<br>- **Code Geass: Lelouch R2**: 21.60<br>- **Fullmetal Alchemist: Brotherhood**: 22.11<br>- **Akame ga Kill**: 22.31 |
| **AVG Distances** | 7.29e-05                                                                                                                                                                   | 0.37                                                                                                                                                                              | 261389.78                                                                                                                                                                              | 20.94                                                                                                                                                                                     |

**Quantitative Feature Comparisons from EXAMPLE 1 (SINGLE INPUT)**

**scored_by** (Mean 51396.646, St.Dev 96648.632)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|2.927|364104.058|
|Cosine|yes|3.482|383189.728|

From the above table for scored_by feature standard deviation, we can see that the scored_by values of Cosine normalized KNN results are on average further from the input average of the scored_by feature compared to the Cosine un-normalized KNN. Our input anime has a high scored value of 1038161. This value of scored_by may have been caused by possible skewing when we normalized our dataset, which may be why normalized KNN has greater variance for large quantitative features as opposed to small quantitative features.

**popularity** (Mean 2988.340, St.Dev 2868.050)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.006|32.422|
|Cosine|yes|0.005|20.449|

From the above table for popularity feature standard deviation, we can see that the popularity values of Cosine un-normalized KNN results are on average further from the input average of the popularity feature compared to the Cosine normalized KNN. This is directly opposite from our feature analysis of scored_by results. However, it should be the popularity of an anime is inversely proportional to its value for the popularity feature. For example, an anime with popularity feature value 4 is mmore popular than an anime with popularity feature value 200. It is likely Cosine normalized KNN performed better than Cosine un-normalized KNN for the popularity feature as our input anime had a popularity of 2, which is a small value and is likely less skewed when normalized.

**episodes** (Mean 18.508, St.Dev 44.939)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|1.295|119.221|
|Cosine|yes|0.284|19.152|

From the above table for episodes feature standard deviation, we can see that the Cosine normalized KNN results had less variance than the Cosine un-normalized results. Similar to the popularity feature results, we expect the normalized KNN results to have less variance as the input episodes value is 25 and within one standard deviation to the mean (less skewed when normalized).

**rank** (Mean 3453.870, St.Dev 2736.869)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.123|598.198|
|Cosine|yes|0.028|83.252|

The results and distribution of the rank feature are similar to that of the popularity feature; a low rank value refers to a high ranking anime, while a high rank value refers to a low ranking anime. Our input rank value was 116 which is a very low rank value compared to the feature distribution (range: 1 to 13837). This is likely why Cosine normalized KNN achieved a lower variance than Cosine un-normalized KNN.

**members** (Mean 100507.587, St.Dev 164257.151)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|2.453|516539.407|
|Cosine|yes|2.466|474466.307|

From the above table for members feature standard deviation, we can see that the members values of Cosine normalized KNN results are on average further from the input average of the members compared to the Cosine un-normalized KNN. This is likely because our input members value was 1500958, a high value that may have been skewed by normalization as the members feature also has a high range (52 to 1610561).

**favorites** (Mean 1610.343, St.Dev 6211.037)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|4.748|31280.509|
|Cosine|yes|5.217|38706.686|

The results for the favorites feature was similar to that of the members feature. Like members, the favorites feature has a large range (0 to 120331) and our input anime had a high favorites value of 70555 (3rd quartile).


 If we compare Average Absolute Standard Z between our quantitative features, favorites had the largest average absolute standard Z. We can expect this, because the favorites feature has a large range of values (from 0 to 120331) and a moderately high variance (6211.037) for its range. Of the features, popularity had the lowest average absolute standard z. Although the range of feature popularity is relatively large (from 1 to 15013), the data distribution for popularity is right-skewed:
 <p align='center'>
  <img src="graphs/popularity distr.png"/>
</p>
and the bulk of the data for popularity is small in value. Because of this distribution, we were able to get results with small variance based on our input popularity, 2. In contrast, if we were to run KNN for an input with larger popularity feature, we would get significantly different results (see below).


**Partial feature test, Median popularity input** (1975)
(Mean 2988.340, St.Dev 2868.050)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.050|180.487|

Our resulting average absolute standard Z for our popularity feature from this test is 0.050, which is much greater than our EXAMPLE 1 test results (average absolute standard Z: 0.006).


EXAMPLE 2, From a single series of anime:
['Attack on Titan', 'Attack on Titan: Since That Day', 'Attack on Titan: Crimson Bow and Arrow', 'Attack on Titan: Wings of Freedom', 'Attack on Titan Season 2', 'Attack on Titan: Junior High', 'Attack on Titan Season 3']
INPUT KEY TAKEAWAY: 'Attack on Titan: Since That Day', 'Attack on Titan: Crimson Bow and Arrow', 'Attack on Titan: Wings of Freedom' 

|               | Cosine Unaltered                                                                                                                                                           | Cosine Normalized                                                                                                                                                                 | Euclidean Unaltered                                                                                                                                                                    |                                                                                    Euclidean Normalized                                                                                   |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 1.11e-16                                                                                                                                                                   | 2.22e-16                                                                                                                                                                          | 0                                                                                                                                                                                      | 0                                                                                                                                                                                         |
| **Distances**     | - **anohana**: 9.24e-06<br>- **Madoka Magica the Movie**: 1.40e-05<br>- **Kuroko's Basketball** 1.42e-05<br>- **Vampire Knight**: 2.51e-05<br>- **Maid Sama!**: 2.68e-05 | - **Gun Samurai Recap**: 0.12<br>- **Marches Comes in Like a Lion**: 0.18<br>- **Berserk: Recollections**: 0.24<br>- **So, I Can't Play H!**: 0.26<br>- **Tsukigakirei: First Half**: 0.31 | - **Miss Kobayashi's Dragon Maid**: 10003.85<br>- **Rosario + Vampire**: 10933.50<br>- **My Teen Romantic Comedy**: 13918.15<br>- **GATE**: 16494.10<br>- **JoJo's Bizarre Adventure**: 18196.80 | - **Marches Comes in Like a Lion**: 21.31<br>- **Persona 4 the Animation**: 27.07<br>- **Fullmetal Alchemist: Premium**: 29.63<br>- **Shiki Specials**: 29.80<br>- **Robot Girls Z**: 30.68 |
| **AVG Distances** | 1.79e-05                                                                                                                                                                   | 0.23                                                                                                                                                                              | 13909.285696612944                                                                                                                                                                              | 27.70                                                                                                                                                                                     |
 

**Quantitative Feature Comparisons from EXAMPLE 2**
 
 **scored_by** (Mean 51396.646, St.Dev 96648.632)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.957|107719.314|
|Cosine|yes|0.431|52104.241|
|Euclidean|yes|0.116|11524.224|
|Euclidean|no|0.760|92020.483|
 
 We can see from the above table for scored_by feature analysis that normalized KNN performed better than un-normalized KNN with regards to our input.
 
 **popularity** (Mean 2988.340, St.Dev 2868.050)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.037|134.361|
|Cosine|yes|0.873|3472.135|
|Euclidean|yes|0.331|1020.332|
|Euclidean|no|0.222|799.213|

In contrast to scored_by results, our popularity feature comparison from EXAMPLE 2 shows that both un-normalized KNN results performed better compared to normalized KNN for popularity.

**episodes** (Mean 18.508, St.Dev 44.939)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.183|9.346|
|Cosine|yes|0.078|4.4|
|Euclidean|yes|0.017|0.894|
|Euclidean|no|0.040|2.332|

With regards to the above table for our episodes feature, both Euclidean KNN results had lower variance from our input episodes feature.

**rank** (Mean 3453.870, St.Dev 2736.869)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.281|967.189|
|Cosine|yes|0.837|2541.881|
|Euclidean|yes|0.654|1908.377|
|Euclidean|no|0.478|1368.623|

For the rank feature, un-normalized KNN results had lower average absoluted standard Z scores in comparison to the normalized KNN results. Cosine un-normalized KNN produced better results than Euclidean un-normalized KNN for the rank feature. However, our Euclidean normalized KNN results had lower variance than our Cosine normalized KNN results.

**members** (Mean 100507.587, St.Dev 164257.151)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.953|180696.498|
|Cosine|yes|0.484|99601.606|
|Euclidean|yes|0.110|18787.324|
|Euclidean|no|0.745|153209.480|

For members, both normalized KNN had improved average absolute standard Z values, opposed to the un-normalized average absolute standard Z scores.

**favorites** (Mean 1610.343, St.Dev 6211.037)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.883|6265.180|
|Cosine|yes|0.035|277.333|
|Euclidean|yes|0.005|38.217|
|Euclidean|no|0.514|639.008|

Like our results for the members feature comparison test, normalized KNN performed better with regards to the favorites feature as well, with Euclidean normalized KNN producing the smallest average absolute standard Z score.

**One-Hot Feature Comparisons from EXAMPLE 2**

For this series of comparisons, the mean value for one-hot feature represents the percentage of the data that has this feature. Some features have relatively high proportions, such as genre_Comedy, which has a mean value of 0.4486 (or 44.86% of the data). In comparison, other features represent a very small percentage of the data, such as studio_Madhouse, which has a mean of 0.0549, representing a 5.49% of the data.
Additionally, we use Absolute average difference as a measure test how similar our results were to the input. It is calculated by:
 <p align='center'>
  <img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%7C%20%5Cfrac%7B%5Csum%20x_%7Bi%7D%7D%7Bn%7D-%5Cmu%20%5Cright%20%7C%20%3D%5Cleft%20%7C%20%5Cbar%7Bx%7D-%5Cmu%20%5Cright%20%7C"/>
</p>
 where x-bar is the average feature value from the anime recommendations and mu is the average feature value from the inputs.
 
 **genre_Action** (Mean 0.3929, St.Dev 0.4885)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|ABS AVG Diff|
|---|---|---|---|---|
|Cosine|no|0.409|0.2|0|
|Cosine|yes|0.409|0.447|0.2|
|Euclidean|yes|0.409|0.2|0.2|
|Euclidean|no|0.655|0.4|0.2|
 
 On average, Cosine un-normalized and both normalized KNN results produced an average absolute standard Z of 0.409, implying those results are more similar to our input series' genre_Action values in comparison to the Euclidean un-normalized results. Euclidean un-normalized KNN performed the "worst", with a average absolute standard Z score of 0.655. From average standard deviaton, we can see that both Cosine un-normalized and Euclidean normalized KNN results had the least average standard feature deviation, with less overall variation from the input series' genre_Action value. However, we cannot use average standard feature deviation as a determining factor for which result was stronger.
 
 **genre_Comedy** (Mean 0.4486, St.Dev 0.4974)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|ABS AVG Diff|
|---|---|---|---|---|
|Cosine|no|0.964|0.489|0|
|Cosine|yes|0.643|0.4|0|
|Euclidean|yes|0.964|0.489|0|
|Euclidean|no|0.964|0.489|0|

In contrast to the genre_Action average absolute standard Z results, Cosine normalized KNN performed the "best", with an average absolute standard Z value of 0.643. Additionally, Cosine normalized KNN results also produced the best average standard feature deviation from the input vector's genre_Comedy value.

**genre_Mystery** (Mean 0.0900, St.Dev 0.2862)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|ABS AVG Diff|
|---|---|---|---|---|
|Cosine|no|1.676|0.489|0|
|Cosine|yes|0|0|0|
|Euclidean|yes|1.676|0.489|0|
|Euclidean|no|0|0|0|

**studio_Madhouse** (Mean 0.0549, St.Dev 0.2280)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|ABS AVG Diff|
|---|---|---|---|
|Cosine|no|0|0|0|
|Cosine|yes|0|0|0|
|Euclidean|yes|0|0|0|
|Euclidean|no|0|0|0|

From our resulting variance measurements, we can see that for one-hot features with very low population represention (small probability), we cannot expect good measurements for how well our recommendations did relative to the input, as most possible data animes fall outside this tiny portion of our data. This is especially exemplified by our measurements from genre_Mystery and studio_Madhouse values for average absolute standard Z and average standard feature deviation; several times, the values were both 0, but this value cannot necessarily signify perfect recommendation results for this feature, given the input anime. Instead, this measurement tells us that from our anime dataset, we do not have enough values in our anime dataset to accurately measure our KNN performance with regards to the feature in question.

In contrast, we found that for one-hot encoded features that are large in proportion (in regards to our anime dataset), Cosine normalized KNN on average performed better than the other KNN implementations. On the other hand, Euclidean un-normlized KNN always performed the worst for such one-hot encoded features. One additional note that should be made here, is that the average standard deviation for one-hot encoded features we were able to measure performance for (namely, genre_Action and genre_Comedy) had average feature standard deviation that approached the overall population standard deviation.
 
 As with the feature comparison trends, overall Cosine un-normalized KNN results prioritized high valued quatitative features over small value features such as one-hot encoded features. In contrast, Cosine normalized KNN produced results that were heavily impacted by one-hot encoded data values like our synopsis encoded data. Seen below is an excerpt of our input synopses:
 <p align='center'>
  <img src="/ML4Anime/graphs/AoT_Series_wording_input.jpg" width="500"/>
</p>
 which heavily featured words like "recap" and "episode." Interestingly, our resulting recommendations from Cosine normalized KNN also produced recommended animes based on these wordings (see below).
 <p align='center'>
  <img src="/ML4Anime/graphs/AoT_Series_wording_input.jpg" width="500"/>
</p>
 Similarly, our Euclidean normalized KNN results also were heavily based on our synopsis key wordings (as seen below):
<p align='center'>
  <img src="/ML4Anime/graphs/AoT_Eu_wording_out-1.jpg" width="500"/>
</p>
  In contrast, our Euclidean un-normalized results were heavily based on high values quantitative features such as scored_by, and did not give results similar to our on-hot encoded features. We can conclude from these results that normalizing our data is imperative to giving equal emphasis to our one-hot features and quantitative data features, but may result in skew due to normalizing high value quantitative feature values.
 

EXAMPLE 3, From a relatively similar assortment of anime:
['Attack on Titan', 'Attack on Titan Season 2', 'Bungo Stray Dogs', 'My Hero Academia 3', 'Nanbaka', 'Nanbaka: Season 2', 'Nanbaka: Idiots with Student Numbers!', 'One Punch Man']
SHARED THEMES/WORDS: survival, human, hero, villain, criminal, police, school, attack

|               | Cosine Unaltered                                                                                                                                                         | Cosine Normalized                                                                                                                                                              | Euclidean Unaltered                                                                                                                                                                           | Euclidean Normalized                                                                                                                                                                                           |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 1.73 e-03                                                                                                                                                                | 0.29                                                                                                                                                                           | 1149911.69                                                                                                                                                                                    | 20.27                                                                                                                                                                                                          |
| **Distances**     | -**Fullmetal Alchemist**: 7.40e-06<br> -**Future Diary**: 9.45e-06<br> -**Elfen Lied**: 9.74e-06<br> -**Parasyte**: 2.14 e-05<br> -**My Teen Romantic Comedy**: 2.59e-05 | -**Fullmetal Alchemist: Brotherhood**: 0.50<br> -**My Hero Academia**: 0.51<br> -**Code Geass: Lelouch**: 0.52<br> -**Death Note**: 0.52<br> -**Code Geass: Lelouch R2**: 0.52 | -**Ouran High School Host Club**: 8961.68<br> -**Maid-Sama!**: 13454.21<br> -**My Teen Romantic Comedy**: 15365.79<br> -**Princess Mononoke**: 18975.94<br> -**Overlord**: 19197.70 | -**JoJo's Bizarre Adventures: Diamond is Unbreakable**: 12.12<br> -**Re:CREATORS**: 12.39<br> -**Akame ga Kill!**: 12.40<br> -**Drifters**: 12.47<br> -**JoJo's Bizarre Adventure: Stardust Crusaders**: 12.76 |
| **AVG Distances** | 1.47e-05                                                                                                                                                                 | 0.52                                                                                                                                                                           | 15191.06                                                                                                                                                                                      | 12.43                                                                                                                                                                                                          |


**Quantitative Feature Comparisons from EXAMPLE 3**
 
 **scored_by** (Mean 51396.646, St.Dev 96648.632)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|1.077|132280.945|
|Cosine|yes|1.662|188732.958|
|Euclidean|yes|1.317|161424.726|
|Euclidean|no|1.411|136816.653|
 
 **popularity** (Mean 2988.340, St.Dev 2868.050)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.013|50.272|
|Cosine|yes|0.002|8.173|
|Euclidean|yes|0.032|119.952|
|Euclidean|no|0.001|2.727|

**episodes** (Mean 18.508, St.Dev 44.939)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.233|13.893|
|Cosine|yes|0.315|17.348|
|Euclidean|yes|0.131|8.634|
|Euclidean|no|0.181|9.410|

**rank** (Mean 3453.870, St.Dev 2736.869)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.128|371.299|
|Cosine|yes|0.015|53.34|
|Euclidean|yes|0.138|421.281|
|Euclidean|no|0.05|162.610|

**members** (Mean 100507.587, St.Dev 164257.151)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|1.032|218850.174|
|Cosine|yes|1.498|272464.600|
|Euclidean|yes|1.196|246220.459|
|Euclidean|no|0.046|8979.482|

**favorites** (Mean 1610.343, St.Dev 6211.037)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.656|4930.025|
|Cosine|yes|4.630|32868.400|
|Euclidean|yes|1.386|9974.895|
|Euclidean|no|0.597|5003.026|


**One-Hot Feature Comparisons from EXAMPLE 3**


 **genre_Comedy** (Mean 0.4486, St.Dev 0.4974)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|ABS AVG Diff|
|---|---|---|---|---|
|Cosine|no|1.045|0.529|0.199|
|Cosine|yes|0.964|0.489|0|
|Euclidean|yes|0.964|0.489|0|
|Euclidean|no|1.45|0.529|0.199|


**genre_Action** (Mean 0.3929, St.Dev 0.4885)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|ABS AVG Diff|
|---|---|---|---|---|
|Cosine|no|0.655|0.4|0|
|Cosine|yes|0.655|0.4|0|
|Euclidean|yes|0|0|0|
|Euclidean|no|0.982|0.489|0|


  ESP FOR GROUPS OF SIMILAR ANIMES, IF INPUT DESCRIPTIONS HAVE OVERLAPPING WORDS, OUTPUT ANIME DESCRIPTIONS HAVE SIMILAR WORDS

EXAMPLE 4, From different anime genres:
['AKIRA', 'Desert Punk', 'Naruto', 'D.N.Angel', 'Rurouni Kenshin']
SHARED THEMES/WORDS: violence, attack, threat, friend, boy, fight, war, Japan, pain, kill

|               | Cosine Unaltered                                                                                                                          | Cosine Normalized                                                                                               | Euclidean Unaltered                                                                                                                              | Euclidean Normalized                                                                                                                                                         |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 8.94e-05                                                                                                                                  | 0.54                                                                                                            | 50161.62                                                                                                                                         | 12.04                                                                                                                                                                        |
| **Distances**     | -**anohana**: 1.02-05<br>-**Parasyte**: 1.36e-05<br>-**Elfen Lied**: 1.36e-05<br>-**Future Diary**: 2.93e-05<br>-**Vampire Knight**: 3.67e-05 | -**Naruto: Shippuden**: 0.51<br>-**Bleach**: 0.53<br>-**Dragonball Z**: 0.54<br>-**Tokyo Ghoul √A**: 0.59<br>-**Reborn!**: 0.59 | -**Haikyu! 2**: 6305.24<br>-**Nisemonogatari**: 10319.20<br>-**School Days**: 12258.90<br>-**Wolf Children**: 12704.43<br>-**Kuroko's Basketball 2**: 12971.85 | -**JoJo's Bizzare Adventure: Stardust Crusaders**: 11.15<br>-**Drifters**: 11.24<br>-**Jojo's Bizarre Adventure**: 11.54<br>-**Evangelion 3.0**: 11.63<br>-**Re:CREATORS**: 11.68 |
| **AVG Distances** | 2.38e-05                                                                                                                                  | 0.55                                                                                                            | 10911.93                                                                                                                                         | 11.45                                                                                                                                                                        |


**Quantitative Feature Comparisons from EXAMPLE 4**
 
 
 **popularity** (Mean 2988.340, St.Dev 2868.050)
 
|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.015|58.669|
|Cosine|yes|0.221|638.734|
|Euclidean|yes|0.30|87.315|
|Euclidean|no|0.191|905.441|

**episodes** (Mean 18.508, St.Dev 44.939)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.135|6.280|
|Cosine|yes|2.971|163.409|
|Euclidean|yes|0.181|9.173|
|Euclidean|no|0.174|8.749|

**favorites** (Mean 1610.343, St.Dev 6211.037)

|DISTANCE|NORMALIZED?|AVG ABS ST.Z|AVG SQ ST.Dev|
|---|---|---|---|
|Cosine|no|0.763|5917.963|
|Cosine|yes|2.348|15738.445|
|Euclidean|yes|0.352|2272.966|
|Euclidean|no|0.257|2129.000|



From our results, we can see for our dataset that on average, Euclidean un-normalized KNN preformed the weakest (highest average output distance). This is likely due to the range of values we have in our dataset. We processed our categorical data into one-hot encoding, as well as retained quantitative values. In comparison, the range and variation of the quantitative values are very high. For example, quatitative feature scored_by has a range from 8 to 1107955, mean of 51396.6469352014, and a standard deviation of 96648.63221428858. Without normalization, using Euclidean distance, which accounts for weight of vectors, as well as the angle between them, will be skewed toward higher values, such as scored_by. In contrast, Cosine un-normalized KNN did a better job for considering quantiative data features.

However, to properly take in our NLP one-hot encoded synopsis data, we should use normalized KNN for better results. This accuracy is improved when a set input anime have closely overlapping or related words. For instance, from our EXAMPLE 3 Cosine normalized KNN test, the input anime synopses shared words like "human", "hero", "villain", "criminal", "fight", and "school". In comparison, the corresponding anime recommendations featured words also featured related words, such as "human", "killer", "hero", "school", "criminal", "vigilante". However, this also has its own downfalls, as quantitative values and one-hot encodes data are normalized to even their weights, more recommendations become heavily dependent on one-hot data. For example, in EXAMPLE 2, specifically the Cosine normalized KNN test, the input anime series (Attack on Titan) had many unrelated but repeating words, such as "recap", "rewrite", "episode", "humanity" and especially contained the phrase "recap of episodes". Likewise, the synopses of the output animes contained this phrase "recap of episode" or a similar variant, but the recommendations were more dependent on this particular synopsis wording, rather than other features.

Additionally, we found that for very different input animes, like in our EXAMPLE 4 test, the KNN recommendations would have higher variance
 - HIGH VARIANCE INPUT DATA (TO MAKE AVERAGE) RESULTS IN HIGHER VARIANCE / MORE SPREAD OUT RECOMMENDATIONS


## Conclusion


Though this approach yielded interesting results, there are some aspects that could be improved. For instance, our current dataset separates out different animes within the same series. Therefore, it could recommend a user who inputs an anime in the series, another anime within the same series. This is obviously not an ideal outcome because avid anime watchers likely would not be getting anything meaningful out of the recommendation engine. Rather, we want to be able to introduce people to new anime that they otherwise might not have known of. One way to address this issue is to compress all of the animes in a series down to one row which would completely eliminate the possibility of these types of results. We could also introduce random noise or uncertainty, not only to mitigate this problem but also so that the results are more likely to be new and interesting to the users. 


### References

[1] Ellis, Theo J. "How the Anime Industry Has Grown Since 2004, According to Google Trends." _Anime Motivation_, animemotivation.com, 23 June 2018, https://animemotivation.com/anime-industry-growth-2004-to-2018/.      

[2] Ellis, Theo J. "Why The Coronavirus Has Made Anime More Popular Than Ever." _Anime Motivation_, animemotivation.com, 24 March 2020, https://animemotivation.com/coronavirus-has-made-anime-more-popular/.      

[3] Mock, Thomas. "Anime Dataset." _GitHub_, GitHub, Inc., 22 April 2019, https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-04-23.

[4] "Full-Text Stopwords." _MySQL_, Oracle Corporation, https://dev.mysql.com/doc/refman/8.0/en/fulltext-stopwords.html.
