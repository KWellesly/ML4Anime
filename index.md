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

EXAMPLE 1, From a single anime title: ['Attack on Titan']
Cosine:
 - unaltered:
    - std dist of inputs: 1.1102230246251565e-16 = ~0
    - dists = (SAO) 4.530671473179648e-05, (DBall Z) 4.819235587816273e-05, (CG: Lelouch R2) 5.287712560930746e-05, (Death Note) 5.83510934309972e-05, (One Punch) 0.00015996019294139963
    - avg dist = 7.29374965183327e-05
 - normalized:
    - std dist of inputs: 2.220446049250313e-16 = ~0
    - dists = (AoT2) 0.26543022986850695, (FullmetalA:B) 0.3679197969434186, (Death Note) 0.3889958931658123, (CG: Lelouch) 0.4021520815895975, (CG: Lelouch R2) 0.44321354005664404
    - avg dist = 0.37354230832479585
Euclidean:
 - unaltered:
    - std dist of inputs: 0
    - dists = (SAO) 68802.63893112267, (Death Note) 132434.60037360518, (FullmetalA:B) 261364.26396249462, (One Punch) 384929.0819958142, (TokyGh) 459418.3602375047
    - avg dist = 261389.7891001083
 - normalized:
    - std dist of inputs: 0
    - dists = (AoT2) 17.510729395599363, (CG: Lelouch) 21.16861555221369, (CG: Lelouch R2) 21.6011141401997, (FullmetalA:B) 22.11454385187206, (AkagaKill) 22.316987243582993
    - avg dist = 20.94239803669356

EXAMPLE 2, From a single series of anime:
['Attack on Titan', 'Attack on Titan: Since That Day', 'Attack on Titan: Crimson Bow and Arrow', 'Attack on Titan: Wings of Freedom', 'Attack on Titan Season 2', 'Attack on Titan: Junior High', 'Attack on Titan Season 3']
INPUT KEY TAKEAWAY: 'Attack on Titan: Since That Day', 'Attack on Titan: Crimson Bow and Arrow', 'Attack on Titan: Wings of Freedom' HAVE SYNOPSIS KEY WORD RECAP
Cosine:
 - unaltered:
    - std dist of inputs: 0.002418394815451697
    - dists = (anohana) 9.243968509209388e-06, (Madoka Movie:Rebellion) 1.4041475445925045e-05, (KnB) 1.4216072585004902e-05, (VampKnight) 2.5147262486702182e-05, (MaidSama) 2.685434382887486e-05
    - avg dist = 1.7900624571143276e-05
 - normalized:
    - std dist of inputs: 0.7088001950884713
    - dists = (Gun Sam Recap) 0.12449084087983442, (Mar Comes in Lion, Recap) 0.1873596877060505, (Berserk: RW, Recap) 0.24814293848774793, (Can't Play H, Recap) 0.26447939233367723, (Tsukigakirei:FH Roadsofar, Recap) 0.3143294973705817
         - ALL RESULTS HAD SYNOPSIS KEY WORD RECAP
    - avg dist = 0.22776047135557836
Euclidean:
 - unaltered:
    - std dist of inputs: 1339680.4229659091
    - dists = 10003.85877635081, 10933.50047045591, 13918.156580823528, 16494.108909603467, 18196.803745831003
    - avg dist = 13909.285696612944
 - normalized:
    - std dist of inputs: 33.11076563385372
    - dists = (Marches Comes in, Recap) 21.316819634794932, (P4, Recap) 27.079444891045924, (FullmetalA: Prem) 29.639697080204925, (Shiki Spec) 29.800487426576534, (Robot Girls Z) 30.68676447637702
    - avg dist = 27.704642701799866
 NORMALIZED WEIGHTED TOWARD SYNOPSIS WORDING, ESP SINCE MANY INPUTS EMPHASIZED SAME WORDS (ESP Recap, episode, member, team)

EXAMPLE 3, From a relatively similar assortment of anime:
['Attack on Titan', 'Attack on Titan Season 2', 'Bungo Stray Dogs', 'My Hero Academia 3', 'Nanbaka', 'Nanbaka: Season 2', 'Nanbaka: Idiots with Student Numbers!', 'One Punch Man']
SHARED THEMES: survival, human, hero, villain, criminal, police, school, attack
Cosine:
 - unaltered:
    - std dist of inputs: 0.0017375156554714224
    - dists = (FA) 7.402229651454206e-06, (FutureDiary) 9.449234395830786e-06, (Elfen Lied) 9.735515303366249e-06, (parasyte) 2.1410417397671466e-05, (My Teen RomCom) 2.5915295883027767e-05
    - avg dist = 1.4782538526270095e-05
 - normalized:
    - std dist of inputs: 0.2960341531481663
    - dists = (FA:B) 0.5021160342395171, (BNHA) 0.5136357132006636, (CG: Lelouch) 0.5229063917691104, (Death Note) 0.5254783458286789, (CG: Lelouch R2) 0.5267359451522967
    - avg dist = 0.5181744860380534
Euclidean:
 - unaltered: 1149911.695198837
    - std dist of inputs:
    - dists = (OuranHost) 8961.677379548775, (MaidSama) 13454.206774389815, (My Teen RomCom) 15365.794548705582, (Princess Mononoke) 18975.944106107854, (Overlord) 19197.709996068443
    - avg dist = 15191.066560964095
 - normalized:
    - std dist of inputs: 20.277913908002063
    - dists = (JoJo:diamond) 12.126737693135617, (Re:CREATORS) 12.391090152953039, (AkagaKill) 12.403364654252414, (Drifters) 12.473460788797862, (JoJo:Stardust) 12.762186996451366
    - avg dist = 12.431368057118059
  ESP FOR GROUPS OF SIMILAR ANIMES, IF INPUT DESCRIPTIONS HAVE OVERLAPPING WORDS, OUTPUT ANIME DESCRIPTIONS HAVE SIMILAR WORDS

EXAMPLE 4, From different anime genres:
['AKIRA', 'Desert Punk', 'Naruto', 'D.N.Angel', 'Rurouni Kenshin']
THEMES: violence, attack, threat, friend, boy, fight, war, Japan, pain, kill
Cosine:
 - unaltered:
    - std dist of inputs: 8.944825003198709e-05
    - dists = (anohana) 1.0174170220200729e-05, (parasyte) 1.365511339768144e-05, (elfen lied) 1.365511339768144e-05, (futdiary) 2.9363935019621756e-05, (VampKn) 3.679790838173602e-05
    - avg dist = 2.3868724349518367e-05
 - normalized:
    - std dist of inputs: 0.5418947476198352
    - dists = (Nar:Shipp) 0.5164024347689984, (Bleach) 0.5350601782020765, (DBall Z) 0.5434340748772342, (TokyGhA) 0.5957931835361809, (Reborn!) 0.5970884948336466
    - avg dist = 0.5575556732436272
   SIMILARITIES IN GENRE??? ESP ACTION, MAGIC
Euclidean:
 - unaltered:
    - std dist of inputs: 50161.62563486801
    - dists = (HQ 2) 6305.243367556172, (Nisemonogatari) 10319.209978109777, (School Day) 12258.909680448818, (WolfChil) 12704.433826394961, (KnB 2) 12971.85888178294
    - avg dist = 10911.931146858533
 - normalized:
    - std dist of inputs: 12.048349500982189
    - dists = (JoJo:Star) 11.159525380754072, (Drifters) 11.248731517957328, (JoJo) 11.547824097220207, (Evangelion:3) 11.63044258371916, (Re:CREATORS) 11.683175678594893
    - avg dist = 11.453939851649132
  RETURNED THEMES: friend, pain, attack, threat, battle, bloody, kill

UNALTERED, EUCLIDEAN ALWAYS SKEWED TOWARD MAINSTREAM VALUES, ESP IF INCLUDE A MAINSTREAM ANIME (large scored_by count)

KEY TAKEAWAYS FROM RESULTS:
 - UNALTERED DATA GETS SKEWED TOWARD VERY LARGE DATA FEATURES SUCH AS scored_by
 - NORMALIZED DATA INCLUDES MORE OVERALL INFORMATION (ESP FOR SYNOPSIS ANALYSIS IF A WORD IS USED OFTEN)
    - POSITIVES: FOR A CLOSE SET OF INPUT ANIME, A RELATED SET OF WORDS SUCH AS (survival, human, hero, villain, criminal, police, school, attack) RESULT IN A RECOMMENDATIONS SET OF CLOSELY ALIGNED THEMES (school, crime, attack, hero, human, fight, revolution)
    - NEGATIVES: FOR A SET OF INPUT ANIME WITH VERY MINIMAL VARIATION IN INPUT (EXACTLY THE THE SAME FEW WORDS ESP), LIKE IN EXAMPLE 2: AoT SERIES HAD MANY SYNOPSES THAT WERE JUST "RECAP OF EPISODES some_range" AND AS EXPECTED, WE WERE RECOMMENDED ANIME WITH SYNOPSES THAT (ESP IN NORMALIZED COSINE) ALL HAD THE WORD 'RECAP' OR 'EPISODE' IN THE DESCRIPTIONS
 - HIGH VARIANCE INPUT DATA (TO MAKE AVERAGE) RESULTS IN HIGHER VARIANCE / MORE SPREAD OUT RECOMMENDATIONS


## Conclusion

[summary of results/things learned].

Though this approach yielded interesting results, there are some aspects that could be improved. For instance, our current dataset separates out different animes within the same series. Therefore, it could recommend a user who inputs an anime in the series, another anime within the same series. This is obviously not an ideal outcome because avid anime watchers likely would not be getting anything meaningful out of the recommendation engine. Rather, we want to be able to introduce people to new anime that they otherwise might not have known of. One way to address this issue is to compress all of the animes in a series down to one row which would completely eliminate the possibility of these types of results. We could also introduce random noise, not only to mitigate this problem but also so that the results are more likely to be new and interesting to the users. 


### References

[1] Ellis, Theo J. "How the Anime Industry Has Grown Since 2004, According to Google Trends." _Anime Motivation_, animemotivation.com, 23 June 2018, https://animemotivation.com/anime-industry-growth-2004-to-2018/.      

[2] Ellis, Theo J. "Why The Coronavirus Has Made Anime More Popular Than Ever." _Anime Motivation_, animemotivation.com, 24 March 2020, https://animemotivation.com/coronavirus-has-made-anime-more-popular/.      

[3] Mock, Thomas. "Anime Dataset." _GitHub_, GitHub, Inc., 22 April 2019, https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-04-23.

[4] "Full-Text Stopwords." _MySQL_, Oracle Corporation, https://dev.mysql.com/doc/refman/8.0/en/fulltext-stopwords.html.
