# Anime Recommendation Engine

## Motivation
Anime is a form of animated media with origins tied to Japan. A recent Google trend revealed that there are between 10-100M searches for anime related topics every month [cite source]. This number has only just peaked in the month of April [cite source] as a result of nation-wide quarantine orders and subsequent efforts to find an entertainment medium. Our goal is to apply machine learning to recommend the best anime for a user to watch based on their personal favorites. Recommendation engines can be built using the techniques of either collaborative or content-based filtering. Due to the limitations of our dataset, our implementation involved using content-based filtering with a modified KNN. To enhance the model and provide only the best of recommendations, we used a combination of dense, categorical, and textual features.

## Data
### Dataset [explain what features we have, what they represent, graphs, etc]
...
### Pre-processing [techniques we used, cleaning text, one-hot encoding, normalizing, graphs, correlation matrix, word embeddings, talk about correlations, etc]
<p><center>
  <img src="https://github.com/KWellesly/ML4Anime/blob/master/graphs/stats_corr_matrix.jpg" width="500"/>
</center></p>
<p><center>
  <img src="https://github.com/KWellesly/ML4Anime/blob/master/graphs/stats_corr_matrix.jpg" width="500"/>
</center></p>
<p><center>
  <img src="https://github.com/KWellesly/ML4Anime/blob/master/graphs/stats_genre_corr_matrix.jpg" width="500"/>
</center></p>
<p><center>
  <img src="https://github.com/KWellesly/ML4Anime/blob/master/graphs/stats_producecr_corr_matrix.jpg" width="500"/>
</center></p>


## Modelling & Results
### Modelling [average of the vector representation of each anime, what distance metric was used, etc]
The KNN algorithm seeks to find the k most similar anime to the current anime. However, often times it is very difficult for users to be able to capture the full breadth of their anime preferences in a single anime. In our modified KNN algorithm, we allow users to input an arbitrary amount of anime that they like in an attempt to better understand and recommend anime catered to their preference. Assume a user inputs *n* different anime that they enjoyed. To model this, we average out the *n* feature vectors of each of those anime and compute KNN on this new vector that ideally captures the essence of each of their preferred animes.
<p><center>
  <img src="https://github.com/KWellesly/ML4Anime/blob/master/graphs/KNN_input_vector.jpg" width="500"/>
</center></p>

<p><center>
  <img src="https://github.com/KWellesly/ML4Anime/blob/master/graphs/KNN_input.jpg" width="500"/>
</center></p>



### Results [show results of KNN before normalizing/PCA, then after KNN on normalized or PCA'd dataset, show examples of results, no way to validate results]
<p><center>
  <img src="https://github.com/KWellesly/ML4Anime/blob/master/graphs/PCA-2D.jpg" width="500"/>
</center></p>

## Conclusion

## References
