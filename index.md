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

### Pre-processing [techniques we used, cleaning text, one-hot encoding, normalizing, graphs, correlation matrix, word embeddings, talk about correlations, etc]

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
### Modelling [average of the vector representation of each anime, what distance metric was used, etc]
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

### Results [show results of KNN before normalizing/PCA, then after KNN on normalized or PCA'd dataset, show examples of results, no way to validate results] --> DON'T PANIC, IS WRITTEN SO IT DOESN'T SOUND LIKE A DRUNK STATISTICIAN

EXAMPLE 1: From a single anime title: ['Attack on Titan']

|               | Cosine Unaltered                                                                                                                                                           | Cosine Normalized                                                                                                                                                                 | Euclidean Unaltered                                                                                                                                                                    |                                                                                    Euclidean Normalized                                                                                   |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 1.11e-16                                                                                                                                                                   | 2.22e-16                                                                                                                                                                          | 0                                                                                                                                                                                      | 0                                                                                                                                                                                         |
| **Distances**     | - **Sword Art Online**: 4.53e-05<br>- **Dragon Ball Z**: 4.82e-05<br>- **Code Geass**: Lelouch R2: 5.28e-05<br>- **Death Note**: 5.83e-05<br>- **One Punch Man**: 1.59e-04 | - **Attack on Titan S2**: 0.26<br>- **Fullmetal Alchemist: Brotherhood**: 0.36<br>- **Death Note**: 0.38<br>- **Code Geass: Lelouch**: 0.40<br>- **Code Geass: Lelouch R2**: 0.44 | - **Sword Art Online**: 68802.63<br>- **Death Note**: 132434.60<br>- **Fullmetal Alchemist: Brotherhood**: 261364.26<br>- **One Punch Man**: 384929.08<br>- **Tokyo Ghoul**: 459418.36 | - **Attack on Titan S2**: 17.51<br>- **Code Geass: Lelouch**: 21.16<br>- **Code Geass: Lelouch R2**: 21.60<br>- **Fullmetal Alchemist: Brotherhood**: 22.11<br>- **Akame ga Kill**: 22.31 |
| **AVG Distances** | 7.29e-05                                                                                                                                                                   | 0.37                                                                                                                                                                              | 261389.78                                                                                                                                                                              | 20.94                                                                                                                                                                                     |

**Quantitative Feature Comparisons from EXAMPLE 1 (SINGLE INPUT)**

**scored_by** (Mean 51396.6469352014, St.Dev 96648.63221428858)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|56588.2|364104.0584099551|
|Cosine|yes|67307.24|383189.72831144626|

**popularity** (Mean 2988.3401050788093, St.Dev 2868.050739389625)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|3.84|32.42221460665511|
|Cosine|yes|2.92|20.449938875214272|

**episodes** (Mean 18.50858143607706, St.Dev 44.939364036423385)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|11.64|119.22164233057687|
|Cosine|yes|2.56|19.15202339179858|

**rank** (Mean 3453.8707530647985, St.Dev 2736.869440698026)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|67.52000000000001|598.1989635564408|
|Cosine|yes|15.479999999999999|83.25262758616091|

**members** (Mean 100507.58774080561, St.Dev 164257.15112195478)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|80586.4|516539.4072722816|
|Cosine|yes|81012.24|474466.3075389021|

**favorites** (Mean 1610.3432574430824, St.Dev 6211.037964762604)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|5898.76|31280.509391632353|
|Cosine|yes|6481.640000000001|38706.68688224297|


EXAMPLE 2, From a single series of anime:
['Attack on Titan', 'Attack on Titan: Since That Day', 'Attack on Titan: Crimson Bow and Arrow', 'Attack on Titan: Wings of Freedom', 'Attack on Titan Season 2', 'Attack on Titan: Junior High', 'Attack on Titan Season 3']
INPUT KEY TAKEAWAY: 'Attack on Titan: Since That Day', 'Attack on Titan: Crimson Bow and Arrow', 'Attack on Titan: Wings of Freedom' HAVE SYNOPSIS KEY WORD RECAP

|               | Cosine Unaltered                                                                                                                                                           | Cosine Normalized                                                                                                                                                                 | Euclidean Unaltered                                                                                                                                                                    |                                                                                    Euclidean Normalized                                                                                   |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 1.11e-16                                                                                                                                                                   | 2.22e-16                                                                                                                                                                          | 0                                                                                                                                                                                      | 0                                                                                                                                                                                         |
| **Distances**     | - **Sword Art Online**: 4.53e-05<br>- **Dragon Ball Z**: 4.82e-05<br>- **Code Geass**: Lelouch R2: 5.28e-05<br>- **Death Note**: 5.83e-05<br>- **One Punch Man**: 1.59e-04 | - **Attack on Titan S2**: 0.26<br>- **Fullmetal Alchemist: Brotherhood**: 0.36<br>- **Death Note**: 0.38<br>- **Code Geass: Lelouch**: 0.40<br>- **Code Geass: Lelouch R2**: 0.44 | - **Sword Art Online**: 68802.63<br>- **Death Note**: 132434.60<br>- **Fullmetal Alchemist: Brotherhood**: 261364.26<br>- **One Punch Man**: 384929.08<br>- **Tokyo Ghoul**: 459418.36 | - **Attack on Titan S2**: 17.51<br>- **Code Geass: Lelouch**: 21.16<br>- **Code Geass: Lelouch R2**: 21.60<br>- **Fullmetal Alchemist: Brotherhood**: 22.11<br>- **Akame ga Kill**: 22.31 |
| **AVG Distances** | 7.29e-05                                                                                                                                                                   | 0.37                                                                                                                                                                              | 261389.78                                                                                                                                                                              | 20.94                                                                                                                                                                                     |
    
 NORMALIZED WEIGHTED TOWARD SYNOPSIS WORDING, ESP SINCE MANY INPUTS EMPHASIZED SAME WORDS (ESP Recap, episode, member, team)
 - ALL RESULTS HAD SYNOPSIS KEY WORD RECAP
 
**Quantitative Feature Comparisons from EXAMPLE 2 (SERIES INPUT)**

**scored_by** (Mean 51396.6469352014, St.Dev 96648.63221428858)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|18512.48|107719.31402492312|
|Cosine|yes|8331.84|52104.241961667576|
|Euclidean|yes|2250.4639999999995|11524.22493012003|
|Euclidean|no|14700.0|92020.48336973676|

**popularity** (Mean 2988.3401050788093, St.Dev 2868.050739389625)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|21.456|134.36160165761646|
|Cosine|yes|501.24799999999993|3472.135746194264|
|Euclidean|yes|190.304|1020.3320243920602|
|Euclidean|no|127.87200000000003|799.2132631532087|

**episodes** (Mean 18.50858143607706, St.Dev 44.939364036423385)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|1.6479999999999997|9.346657156438337|
|Cosine|yes|0.7040000000000001|4.4|
|Euclidean|yes|0.16|0.8944271909999159|
|Euclidean|no|0.36800000000000005|2.33238075793812|

**rank** (Mean 3453.8707530647985, St.Dev 2736.869440698026)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|no|Cosine|154.176|967.1896194645599|
|yes|Cosine|458.52799999999996|2541.881696696367|
|yes|Euclidean|358.15999999999997|1908.3776355847394|
|no|Euclidean|262.15999999999997|1368.6231037067876|

**members** (Mean 100507.58774080561, St.Dev 164257.15112195478)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|31336.432|180696.49885418368|
|Cosine|yes|15927.472|99601.60660069695|
|Euclidean|yes|3645.4080000000004|18787.324007425857|
|Euclidean|no|24489.951999999997|153209.48038629984|

**favorites** (Mean 1610.3432574430824, St.Dev 6211.037964762604)

|DISTANCE|NORMALIZED?|AVG ST.Z|AVG SQ ST.Z|
|---|---|---|---|
|Cosine|no|1097.76|6265.180795475897|
|Cosine|yes|44.368|277.3334455127978|
|Euclidean|yes|7.312|38.2172735814579|
|Euclidean|no|639.008|639.008|


EXAMPLE 3, From a relatively similar assortment of anime:
['Attack on Titan', 'Attack on Titan Season 2', 'Bungo Stray Dogs', 'My Hero Academia 3', 'Nanbaka', 'Nanbaka: Season 2', 'Nanbaka: Idiots with Student Numbers!', 'One Punch Man']
SHARED THEMES: survival, human, hero, villain, criminal, police, school, attack

|               | Cosine Unaltered                                                                                                                                                         | Cosine Normalized                                                                                                                                                              | Euclidean Unaltered                                                                                                                                                                           | Euclidean Normalized                                                                                                                                                                                           |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 1.73 e-03                                                                                                                                                                | 0.29                                                                                                                                                                           | 1149911.69                                                                                                                                                                                    | 20.27                                                                                                                                                                                                          |
| **Distances**     | -**Fullmetal Alchemist**: 7.40e-06<br> -**Future Diary**: 9.45e-06<br> -**Elfen Lied**: 9.74e-06<br> -**Parasyte**: 2.14 e-05<br> -**My Teen Romantic Comedy**: 2.59e-05 | -**Fullmetal Alchemist: Brotherhood**: 0.50<br> -**My Hero Academia**: 0.51<br> -**Code Geass: Lelouch**: 0.52<br> -**Death Note**: 0.52<br> -**Code Geass: Lelouch R2**: 0.52 | -**Ouran High School Host Club**: 8961.68<br> -**Kaichou Wa Maid-Sama**: 13454.21<br> -**My Teen Romantic Comedy**: 15365.79<br> -**Princess Mononoke**: 18975.94<br> -**Overlord**: 19197.70 | -**JoJo's Bizarre Adventures: Diamond is Unbreakable**: 12.12<br> -**Re: CREATORS**: 12.39<br> -**Akame ga Kill**: 12.40<br> -**Drifters**: 12.47<br> -**JoJo's Bizarre Adventure: Stardust Crusadors**: 12.76 |
| **AVG Distances** | 1.47e-05                                                                                                                                                                 | 0.52                                                                                                                                                                           | 15191.06                                                                                                                                                                                      | 12.43                                                                                                                                                                                                          |

  ESP FOR GROUPS OF SIMILAR ANIMES, IF INPUT DESCRIPTIONS HAVE OVERLAPPING WORDS, OUTPUT ANIME DESCRIPTIONS HAVE SIMILAR WORDS

EXAMPLE 4, From different anime genres:
['AKIRA', 'Desert Punk', 'Naruto', 'D.N.Angel', 'Rurouni Kenshin']
THEMES: violence, attack, threat, friend, boy, fight, war, Japan, pain, kill

|               | Cosine Unaltered                                                                                                                          | Cosine Normalized                                                                                               | Euclidean Unaltered                                                                                                                              | Euclidean Normalized                                                                                                                                                         |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **STD Input Distance**  | 8.94e-05                                                                                                                                  | 0.54                                                                                                            | 50161.62                                                                                                                                         | 12.04                                                                                                                                                                        |
| **Distances**     | -**Anohana**: 1.02-05<br>-**Parasyte**: 1.36e-05<br>-**Elfen Lied**: 1.36e-05<br>-**Future Diary**: 2.93e-05<br>-**Vampire Knights**: 3.67e-05 | -**Naruto Shippuden**: 0.51<br>-**Bleach**: 0.53<br>-**Dragonball Z**: 0.54<br>-**Tokyo Ghoul**: 0.59<br>-**Reborn!**: 0.59 | -**HQ 2**: 6305.24<br>-**Nisemonogatari**: 10319.20<br>-**School Day**: 12258.90<br>-**Wolf Children**: 12704.43<br>-**Kuroko no Basket 2**: 12971.85 | -**JoJo's Bizzare Adventure: Stardust Crusaders**: 11.15<br>-**Drifters**: 11.24<br>-**Jojo's Bizarre Adventure**: 11.54<br>-**Evangelion 3.0**: 11.63<br>-**Re:CREATORS**: 11.68 |
| **AVG Distances** | 2.38e-05                                                                                                                                  | 0.55                                                                                                            | 10911.93                                                                                                                                         | 11.45                                                                                                                                                                        |

From our results, we can see for our dataset that on average, Euclidean un-normalized KNN preformed the weakest (highest average output distance). This is likely due to the range of values we have in our dataset. We processed our categorical data into one-hot encoding, as well as retained quantitative values. In comparison, the range and variation of the quantitative values are very high. For example, quatitative feature scored_by has a range from 8 to 1107955, mean of 51396.6469352014, and a standard deviation of 96648.63221428858. Without normalization, using Euclidean distance, which accounts for weight of vectors, as well as the angle between them, will be skewed toward higher values, such as scored_by. In contrast, Cosine un-normalized KNN did a better  
UNALTERED, EUCLIDEAN ALWAYS SKEWED TOWARD MAINSTREAM VALUES, ESP IF INCLUDE A MAINSTREAM ANIME (large scored_by count)

KEY TAKEAWAYS FROM RESULTS:
 - UNALTERED DATA GETS SKEWED TOWARD VERY LARGE DATA FEATURES SUCH AS scored_by
 - NORMALIZED DATA INCLUDES MORE OVERALL INFORMATION (ESP FOR SYNOPSIS ANALYSIS IF A WORD IS USED OFTEN)
    - POSITIVES: FOR A CLOSE SET OF INPUT ANIME, A RELATED SET OF WORDS SUCH AS (survival, human, hero, villain, criminal, police, school, attack) RESULT IN A RECOMMENDATIONS SET OF CLOSELY ALIGNED THEMES (school, crime, attack, hero, human, fight, revolution)
    - NEGATIVES: FOR A SET OF INPUT ANIME WITH VERY MINIMAL VARIATION IN INPUT (EXACTLY THE THE SAME FEW WORDS ESP), LIKE IN EXAMPLE 2: AoT SERIES HAD MANY SYNOPSES THAT WERE JUST "RECAP OF EPISODES some_range" AND AS EXPECTED, WE WERE RECOMMENDED ANIME WITH SYNOPSES THAT (ESP IN NORMALIZED COSINE) ALL HAD THE WORD 'RECAP' OR 'EPISODE' IN THE DESCRIPTIONS
 - HIGH VARIANCE INPUT DATA (TO MAKE AVERAGE) RESULTS IN HIGHER VARIANCE / MORE SPREAD OUT RECOMMENDATIONS


## Conclusion


Though this approach yielded interesting results, there are some aspects that could be improved. For instance, our current dataset separates out different animes within the same series. Therefore, it could recommend a user who inputs an anime in the series, another anime within the same series. This is obviously not an ideal outcome because avid anime watchers likely would not be getting anything meaningful out of the recommendation engine. Rather, we want to be able to introduce people to new anime that they otherwise might not have known of. One way to address this issue is to compress all of the animes in a series down to one row which would completely eliminate the possibility of these types of results. We could also introduce random noise or uncertainty, not only to mitigate this problem but also so that the results are more likely to be new and interesting to the users. 


### References

[1] Ellis, Theo J. "How the Anime Industry Has Grown Since 2004, According to Google Trends." _Anime Motivation_, animemotivation.com, 23 June 2018, https://animemotivation.com/anime-industry-growth-2004-to-2018/.      

[2] Ellis, Theo J. "Why The Coronavirus Has Made Anime More Popular Than Ever." _Anime Motivation_, animemotivation.com, 24 March 2020, https://animemotivation.com/coronavirus-has-made-anime-more-popular/.      

[3] Mock, Thomas. "Anime Dataset." _GitHub_, GitHub, Inc., 22 April 2019, https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-04-23.

[4] "Full-Text Stopwords." _MySQL_, Oracle Corporation, https://dev.mysql.com/doc/refman/8.0/en/fulltext-stopwords.html.
