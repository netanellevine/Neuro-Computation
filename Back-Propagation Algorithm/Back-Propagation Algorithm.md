**Back-Propagation Algorithm Neural Networks** 

- **Matan Yarin Shimon: 314669342** 
- **Or Yitshak: 208936039** 
- **Netanel Levine: 312512619** 
- **Yahalom Chasid: 208515577**  

**Dataset:** 

Our dataset holds 1000 2D points ( , ) and each point has a value that will mark it as  .

- The value of  is of the form ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.001.png)  where  is an integer,  −10000 ≤ ≤ 10000. 100
- The value of  is of the form ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.002.png)  where  is an integer,  −10000 ≤ ≤ 10000. 100
- The value of  is −1  1. 

As a reminder in part A, the value of  was determined by: 

1, > 1

- (( , )) = {

−1, ℎ

As a reminder in part B, the value of  was determined by: 

1, 4 ≤ 2 + 2 ≤ 9

- (( , )) = {

−1, ℎ

**Part C:** 

In this part, we will try to improve our scores from parts A and B by using the Back-Propagation Algorithm with the MLP Classifier. 

In part A our score was:  . %

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.003.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.004.png)

Explanation of the right scatter: 

- The **red** area is all the points that their  = −1 because  <= 1. 
- The **green** area is all the points that their  = 1 because  > 1. 
- The **blue** dots mark the points that our model predicted as  = 1. 
- The **purple** dots mark the points that our model predicted as  = −1. 
1. Every time our model marked a point as **blue** and this point lies in the range of the **green** area, our model predicted the correct answer. 
1. Every time our model marked a point as **purple** and this point lies in the range of the **red** area, our model predicted the correct answer. 
1. Every time our model marked a point as **blue** and this point lies in the range of the **red** area, our model predicted the wrong answer. 
1. Every time our model marked a point as purple and this point lies in the range of the **green** area, our model predicted the wrong answer. 

As we can see, near the line where  = 0 the model had a few mistakes. 

Because our score is almost perfect and Back-Propagation is much more accurate than Adaline we won’t perform the Back-Propagation Algorithm on the part A function. 

In part B we scored  . %** and we thought that this is too good to be true. 

We did some digging and we found out that the range of the points that can get a value of 1 is small if we compare it to the range of all the points. This means a high probability of a point 

( 100%) landing in the −1 range which can cause our model to overfit and  

always predict  −1. 

As we can see in the confusion matrix below, we got amazing result but the model predict most of the time −1: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.005.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.006.png)

So we decided to balance the data by giving our model approximately half instances from  

each type. After doing so our model accuracy was quite bad  . % with Adaline Algorithm. 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.007.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.008.png)Finally, now we will try to improve our model accuracy by using the Back-Propagation Algorithm using the MLP. 

Let’s recall our problem: 

1, 4 ≤

- (( , )) = {

−1, ℎ

2 + 2 ≤ 9

We solved this problem using the MLP Classifier from Sklearn.  

The MLPClassifier uses Feed-Forward Networks and Back-propagation, with a given data and desired output, the MLPClassifier train the model and changes the weight and bias accordingly. 

We chose to do it with 2 hidden layers: 

1. The first hidden layer will have 8 neurons. 
1. The second hidden layer will have 2 neurons. 

Below we can see a diagram for this network:

**Hidden First Layer  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.009.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.010.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.011.png)**

**Input Layer  Hidden Second Layer ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.012.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.013.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.014.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.015.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.016.png)**

**1  5 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.017.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.017.png)**

8 inputs for

` `each neuron![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.018.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.019.png)

**1  1 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)**

**2  6 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)**

**3  7  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.021.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.022.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.023.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.024.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)**

8 inputs for

` `each neuron![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)

**2  2** 

**4  8 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.025.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.026.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)**

**Output Layer  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.027.png) Output Layer After Zoom  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.028.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.029.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.030.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.031.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.032.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.033.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.034.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.035.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.036.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.037.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.038.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.039.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.039.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.040.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.041.png)**

Because our formula is: 

- (( , )) = { 1, 4 ≤ 2 + 2 ≤ 9 −1, ℎ

It makes sense that all the green area will look like the space with: 

All the circles such that: 2 ≤ ≤ 3,  and with the center of (0,0), and with  , such that:  {4 ≤ 2 + 2 ≤ 9}.* 

To conclude the Back-Propagation Algorithm using the MLP result of part B with  the balanced dataset was  . %**.**

Which is almost twice as better as the Adaline score which was  . %**.** We can see from the Confusion Matrix that the model doesn’t overfit: 

**Back-Propagation  Adaline** 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.042.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.043.png)

**Part D:** 

In this part, we will use our trained neurons from the next to last level of part C as our input. The last neuron will do the prediction according to the Adaline algorithm and not the Back- Propagation Algorithm that we used in part C, and we will train only the Adaline neuron. 

Our problem is the same as before:  

1, 4 ≤ 2 + 2 ≤ 9

- (( , )) = {

−1, ℎ

After doing so, combining the MLP Classifier with the Adaline Algorithm, our model  score was  . %.

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.044.png)

Which is almost the same as the result of the Back-Propagation Algorithm  . %**.** Below we can see a diagram for this network:

**Hidden First Layer  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.009.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.010.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.011.png)**

**Input Layer  Hidden Second Layer ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.012.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.013.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.014.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.015.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.016.png)**

**1  5 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.017.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.045.png)**

8 inputs for

` `each neuron![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.018.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.019.png)

**1  1 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)**

**2  6 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)**

**3  7  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.021.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.022.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.023.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.024.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)**

8 inputs for

` `each neuron![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)

**2  2** 

**4  8 ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.025.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.026.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.020.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.046.png)**

**Output Layer  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.027.png) Output Layer After Zoom  ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.028.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.047.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.048.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.049.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.032.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.033.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.034.jpeg)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.035.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.036.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.037.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.038.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.039.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.039.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.041.png)![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.040.png)**

As we saw in the diagram the scatters of the Adaline using the MLP and the Back-Propagation using the MLP are almost identical, which is why both of the scores are almost the same. From the Confusion Matrix, we can conclude that both models’ prediction was quite similar. 

**Adaline with MLP** 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.050.jpeg)

**Back-Propagation with MLP** 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.051.jpeg)

We can see that the only difference between those models is that Adaline with MLP predicted  2  points as −1 while Back-Propagation with MLP predicted apparently those points as 1. 

To conclude, our first Adaline (from part B) score was: **55.879%**, the Back-Propagation with MLP score was **98.28%,** and the Adaline with MLP score was **98.3%**. 

So we saw that MLP was able to improve our model and by using MLP in two different Algorithms we got almost the same scores. 

**Code:** 

Functions to create random points the first is balanced the second is not: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.052.jpeg)

Main: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.053.png)

partC function: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.054.png)

diagram: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.055.png)

show\_area: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.056.jpeg)

get\_cm (confusion matrix): 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.057.png)

partD function: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.058.png)

Adaline\_with\_MLP: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.059.jpeg)

get\_last\_bp\_layer\_for\_adaline: 

![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.060.jpeg)

show\_areaD: ![](Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.061.jpeg)
