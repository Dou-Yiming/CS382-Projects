# Project1: Smoothing Algorithms of n-gram Language Model

## 1. Construction of n-gram Language Model

#### 1.1 Model Design

In order to thoroughly compare the ability of different language models, I constructed **4 models** with different grams in this project, including **unigram, bigram, trigram and 4-gram**. 

n-gram language model is **statistic model**, which means that the training procedure only includes counting the words or phrases in the corpus. 

Take the construction of unigram model as an example (bigram, trigram and 4-gram models are constructed similarly). Before counting the amount of each word, we should begin with **listing all of the different words** (also called "type") that exist in the whole corpus, which is the union of train_set, dev_set and test_set. Next, the number of each type in the unigram model is **initialized as $0$**:

```python
for word in corpus:
    model.unigram[word] = 0
```

Then, we traverse through the train_set and **count the number of each word**, preparing for the following discounting and testing procedure:

```python
for word in train_set:
	model.unigram[word] += 1
```

#### 1.2 Sparsity Problem

After counting each word in the train_set, we notice that some of the words do not exist in train_set, making the count of them 0. This problem is also called **sparsity problem**, which is a serious problem for the following reason.

During the test procedure, the perplexity is computed by the following formula:
$$
PPL=10^{-\frac{1}{K}\sum_{k=1}^Klog_{10}P(\omega_k|W^{k-1}_{k-n+1})}
$$
Obviously, if $P=0$, the PPL will become an **infinite number**, which is not acceptable. 

Therefore, some algorithms have to be designed to solve sparsity problem. There are two kinds of solutions:

1. **Discounting**: To keep a language model from assigning zero probability to these unseen events, we shave off a bit of probability mass from some more frequent events and give it to the events we’ve never seen. The algorithm can be denoted as:
    $$
    P_{discount}(\omega|y)=d(y,\omega)\frac{C(y,\omega)}{C(y)}
    $$
    $d(y,\omega)$ is called "discounting coefficient".

2. **Back-off & Interpolation:** Recursively move back to lower-order n-grams, in the end we will get a robust estimation.

In this project, $3$ discounting algorithms are designed and implemented, which is **Jelinek-Mercer Smoothing**, **Add-k Smoothing** and **Good-Turing Smoothing**, respectively. The detailed design and implementation are shown in the next section.

## 2. Design and Implementation of Discounting Algorithms

In this section, the 3 discounting algorithms are discussed in detail.

#### 2.1 Jelinek-Mercer Smoothing (Interpolation)

- Basic Idea

    Some of the n-grams in the test set may not be included in the train-set, making their probability become $0$. We can estimate the probability of these n-grams by combine different n-grams linearly interpolating all the models.

- Algorithm Design

    For example, if we are willing to estimate trigram probability, the probability can be computed by a weighted sum of trigram, bigram and unigram probabilities:
    $$
    P(\omega_n|\omega_{n-2},\omega_{n-1})=\lambda_1P(\omega_n|\omega_{n-2},\omega_{n-1})+\lambda_2P(\omega_n|\omega_{n-1})+\lambda_3P(\omega_n)
    $$
    We should make sure that $\sum_i\lambda_i=1$.

    Obviously, the formula can be extended to fit the higher-dimension:
    $$
    P(\omega_i|W_{i-n+1}^{i-1})=\lambda_nP(\omega_i|W_{i-n+1}^{i-1})+(1-\lambda_n)P(\omega_i|W_{i-n+2}^{i-1})
    $$
    

- Implementation

    When it comes to the specific implementation of the algorithm, it can be recursively executed by the following code:

    ```python
    num2name = {
        1: 'unigram',
        2: 'bigram',
        3: 'trigram',
        4: 'quagram'
    }
    def prob(model, key, lambd):
    	# end of recursion    
        if len(key) == 1:
            return model['unigram'][key]
    
        model_name = num2name[len(key)]
        if not key in model[model_name].keys():
            return (1 - lambd) * prob(model, tuple(key[1:]), lambd)
        return lambd * model[model_name][key] + (1-lambd) * prob(model, tuple(key[1:]), lambd)
    ```

#### 2.2 Add-k Smoothing

- Basic Idea

    Since there are several unseen cases, the **intuitive way** to do smoothing is to add $1$ to all of the n-gram counts before we normalize them into probabilities. This method is also called Laplace smoothing.

- Algorithm Design

    Suppose there are $V$ words in the vocabulary (the number of different types in the corpus). We also need to adjust the denominator to take the extra $V$ observations into account:
    $$
    P_{Laplace}(\omega_i)=\frac{c_i+1}{N+V}
    $$
    Furthermore, in the practical situation, adding $1$ to every word may be too large. Hence, we may add a fractional count $k$ instead of $1$ to each word:
    $$
    P_{Laplace}(\omega_i)=\frac{c_i+k}{N+kV}
    $$
    

    This is why the method is called add-k smoothing.

- Implementation

    The implementation can be denoted as the following code:

    ```python
    model.unigram = {
        word: (count+k)/(N+k*V) for word, count in model.unigram.items()}
    ```

#### 2.3 Good-Turing Smoothing

- Basic Idea

    **Redistribute** the probability of n-grams that appear for $r+1$ times to those that appear for $r$ times.

- Algorithm Design

    Suppose $N_{r}$ is the "**counts of counts**", which is the amount of n-grams that appear for $r$ times in the train_set. Hence, the amount of observed n-grams in the train_set is
    $$
    N=\sum_{r=0}^{\infty}rN_r
    $$
    Next, the new $r^*$ can be denoted as
    $$
    r^*=(r+1)\frac{N_{r+1}}{N_r}
    $$
    

    Therefore, the new probability of an n-gram can be defined as:
    $$
    P^*=\frac{r^*}{N}
    $$

- Implementation

    Based on the discussion above, we can implement the algorithm by the following code:

    ```python
    for k in model.unigram.keys():
        	r_star = model.unigram[k] + 1
            if r_star in Nr.keys():
                model.unigram[k] = r_star * Nr[r_star] / Nr[r_star - 1]
    ```

## 3. Experiments & PPL Results

In this section, the experiments testing the effects of algorithms are performed.

To compare the ability of models with different grams and test the influence of lambda in interpolation, the perplexity is tested using models ranging from unigram to 4-gram and lambda ranging from $0.05$ to $0.95$.

Notice that **the whole test_set is regarded as a single sentence** in the experiments, thus only one beginning token and one ending token are added to the test_set. Moreover, due to the significant long-tail effect of the dataset that is shown in the figure below ($cnt=log_{10}cnt$), the words that show less than $2$ times in the train-set are replaced with "UNK", resulting in the vocabulary size shrinking to $118849$.

<img src=".\visualization\train_set.jpg" style="zoom:60%;" />

In the **first** experiment, **only the interpolation algorithm is used**, while no discounting is performed.

The perplexity results are shown in the following table:

| Model\lambda |    0.05    |    0.1     |    0.2     |     0.4      |    0.8     |    0.95    |
| :----------: | :--------: | :--------: | :--------: | :----------: | :--------: | :--------: |
| **unigram**  |  1536.56   |  1536.56   |  1536.56   |   1536.56    |  1536.56   |  1536.56   |
|  **bigram**  |   950.82   |   802.16   |   651.93   |    521.21    | **468.98** | **561.97** |
| **trigram**  |   705.02   |   575.16   | **471.54** | ***426.00*** |   704.34   |  1821.56   |
|  **4-gram**  | **647.07** | **542.45** |   483.24   |    547.17    |  2171.33   |  16776.59  |

In the **second** experiment, the **Good-Turing algorithm is performed first**, followed by interpolation algorithm.

The perplexity results are shown in the following table:

| Model\lambda |    0.05    |    0.1     |    0.2     |    0.4     |    0.8     |    0.95    |
| :----------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| **unigram**  |  1536.56   |  1536.56   |  1536.56   |  1536.56   |  1536.56   |  1536.56   |
|  **bigram**  |   980.45   |   834.46   |   685.76   |   556.30   | **512.26** | **620.71** |
| **trigram**  |   768.97   |   639.77   | **536.97** | **500.26** |   873.91   |  2328.53   |
|  **4-gram**  | **725.77** | **623.15** |   571.22   |   672.11   |  2868.68   |  23084.52  |

## 4. Conclusion

Based on the results, we can come to the following conclusions:

1. For smaller value of $\lambda$, models with larger grams performs better. 

    The reason is that when $\lambda$ is small, the influence of sparsity problem is relatively small, making it rely more on interpolated values.

2. To reach the best result of perplexity, the value of $\lambda$ should not be too large or too small. 

    As is shown in the table above, the best PPL result is $426.00$ when $\lambda=0.4$ and trigram model is used.

3. Performing Good-Turing may result in larger perplexity.

    Generally speaking, the model that uses Good-Turing algorithm results in perplexity that is slightly larger than the model that does not use it.

## 5. Reference

1. Slides of SJTU-CS382
2. [N-Gram Language Models | Towards Data Science](https://towardsdatascience.com/n-gram-language-models-af6085435eeb)
3. [Additive smoothing - Wikipedia](https://en.wikipedia.org/wiki/Additive_smoothing)
4. [n-gram - Wikipedia](https://en.wikipedia.org/wiki/N-gram)
5. [cs224n-lecture2-language-models.ppt (stanford.edu)](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf)
6. [Good–Turing frequency estimation - Wikipedia](https://en.wikipedia.org/wiki/Good–Turing_frequency_estimation)
7. [good-turing-smoothing-without.pdf (ntu.edu.tw)](https://www.csie.ntu.edu.tw/~b92b02053/print/good-turing-smoothing-without.pdf)

## 6. Supplementary

1. To **execute the code**, you may simply run the code below in the project folder (parameters can be modified in config file):

    ```shell
    # Train the model and save model parameters to trained_model folder
    python ./main.py --config_path ./configs/default.json
    # Load the trained model and only test perplexity
    python ./main.py --config_path ./configs/test.json
    ```
    
    Packages you may need to install by simply executing `pip install package-name`:

    ```
    numpy
    tqdm
    pprint
    argparse
    easydict
    ```
    
2. **I guarantee that everything of the code and report is done by myself without using any complex open-source tool other than basic packages such as numpy**. 

    If you have any question for any section of this project, please feel free to contact me by the following methods:

    E-mail: [douyiming@sjtu.edu.cn](mailto:douyiming@sjtu.edu.cn)

    Wechat: 18017112986

