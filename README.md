Expand a lexicon with pretrained GloVe embeddings (trained on Tweets)
================

In this tutorial we will download pre-trained word embeddings -
[GloVe](https://nlp.stanford.edu/projects/glove/) - developed by the
Stanford NLP group. In particular, we will use their word vectors
trained on 2 billion tweets. Other versions are available e.g., a model
trained on wikipedia data.

## 1 - Download GloVe word embeddings

``` r
# download pre-trained vectors trained on twitter data
download.file("http://nlp.stanford.edu/data/glove.twitter.27B.zip", destfile = "glove.zip")
unzip("glove.zip", exdir = "E:/Downloads/GLOVE")
```

## 2 - Setup

``` r
# load libraries
library(data.table)
library(text2vec)
library(tictoc)
library(tidyverse)

# set dir
setwd("E:/Downloads/GLOVE")

# load text file of word vectors
g27b_200 <- scan(file = "E:/Downloads/GLOVE/glove.twitter.27B.200d.txt", what="", sep="\n")
```

## 3 - Load some functions we use below

#### Functions to process the pretrained model and use cosine similarity to extract similar words

``` r
# load this function to process pretrained word embeddings
proc_pretrained_vec <- function(pretrained_vec) {
  
  # initialize space for values and the names of each word in vocab
  vals <- vector(mode = "list", length(pretrained_vec))
  names <- character(length(pretrained_vec))
  
  # loop through to gather values and names of each word
  for(i in 1:length(pretrained_vec)) {
    if(i %% 1000 == 0) {print(i)}
    this_vec <- pretrained_vec[i]
    this_vec_unlisted <- unlist(strsplit(this_vec, " "))
    this_vec_values <- as.numeric(this_vec_unlisted[-1])  # this needs testing, does it become numeric?
    this_vec_name <- this_vec_unlisted[1]
    
    vals[[i]] <- this_vec_values
    names[[i]] <- this_vec_name
  }
  
  # convert lists to data.frame and attach the names
  glove <- data.frame(vals)
  names(glove) <- names
  
  return(glove)
}

# load function to find cosine similarity using word vectors, default is seed word + 5 most similar words = 6
find_sim_wvs <- function(word_vector, glove_vectors, top_n_res = 6) {
  # word_vector is a numeric vector; glove_vectors is a data.frame with words as columns and dimesions as rows
  this_wv_mat <- matrix(word_vector, ncol=length(word_vector), nrow = 1)
  all_wvs_mat <- as.matrix(glove_vectors)
  
  if(dim(this_wv_mat)[[2]] != dim(all_wvs_mat)[[2]]) {
    print("switching dimensions on the glove matrix")
    all_wvs_mat <- t(all_wvs_mat)
  }
  
  cos_sim = sim2(x=all_wvs_mat, y=this_wv_mat, method="cosine", norm="l2")
  sorted_cos_sim <- sort(cos_sim[,1], decreasing = T) 
  return(head(sorted_cos_sim, top_n_res))
  
}
```

## 4 - Load GloVe embeddings

``` r
# call the function to make into list
glove.200 <- proc_pretrained_vec(g27b_200)

# we loaded the 2 billion tweet, 200 dimension-per-word, 1193514 vocabulary word vector
print(dim(glove.200)) 
```

## 5 - Define your dictionary

#### We will also check whether there are word vectors available for each word in the dictionary. This is because if not it will stop the loop below.

``` r
# define your dictionary
dict <- c("warming", "2nd", "democrat", "republican", "blame")

# if you find a word with no vector (NULL), you will need to remove the word or else it stops the loop below
for (i in 1:length(dict)){
  if(is.null(glove.200[[dict[i]]]) == TRUE) {
    print(paste("Delete the word:", dict[i]))
  } 
}
```

    ## [1] "Delete the word: 2nd"

``` r
# we find that the phrase "2nd" contains no word vectors, so remove it
dict <- c("warming", "democrat", "republican", "blame")
```

## 6 - Set up similar word search

``` r
# create dataframe to store results
similar_words <- data.frame(matrix(ncol = length(dict), nrow = 6))

# test it out for 1 word vector to get estimate of how long it will take to generate your expanded dict
tic()
this_word_vector <- glove.200[['warming']]      
find_sim_wvs(this_word_vector, glove.200, top_n_res = 6)
```

    ## [1] "switching dimensions on the glove matrix"

    ##       warming       climate        global       heating climatechange 
    ##     1.0000000     0.6737770     0.5464444     0.5309427     0.5253251 
    ##          cold 
    ##     0.5096390

``` r
toc()
```

    ## 22.42 sec elapsed

#### On my machine, generating 5 similar words for 1 word vector takes \~20 secs).

## 7 - Get similar words

``` r
# loop to use function over dictionary, store top 5 related words (set to 6 words because first word is the dict word)
for (i in 1:length(dict)){
  this_word_vector <- glove.200[[dict[i]]]
  cos_sim <- find_sim_wvs(this_word_vector, glove.200, top_n_res = 6)
  similar_words[1:6,i] <- names(cos_sim)
}

# pull the first row and make it the column name
names(similar_words) <- similar_words %>% slice(1) %>% unlist()
similar_words <- similar_words %>% slice(-1)

# export your expanded dictionary
similar_words %>% write_csv("dict_expanded.csv")
```
