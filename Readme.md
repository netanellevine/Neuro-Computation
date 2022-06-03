# Back-Propagation Algorithm Neural Networks

- **Matan Yarin Shimon** 
- **Netanel Levine** 
- **Or Yitshak**
- **Yahalom Chasid**  

##Dataset:  
Our dataset holds 1000 2D points (x,y) and each point has a value that will mark it as **v**.
![](readme_pics/im0.png)

As a reminder in part A, the value of **v** was determined by: 

![](readme_pics/im1.png)

As a reminder in part B, the value of **v** was determined by: 

![](readme_pics/im2.png)

##Part C 

In this part, we will try to improve our scores from parts A and B by using the Back-Propagation Algorithm with the MLP Classifier. 

Part A our score was: <span style="font-size:18.0pt"> **99.5%**</span>

![](readme_pics/im3.png)


Explanation of the right scatter: 
<span style="color: red;">text</span>

- The **<span style="color: red;">red</span>** area is all the points that their  = −1 because  <= 1. 
- The **<span style="color: green;">green</span>** area is all the points that their  = 1 because  > 1. 
- The **<span style="color: blue;">blue</span>** dots mark the points that our model predicted as  = 1. 
- The **<span style="color: purple;">purple</span>** dots mark the points that our model predicted as  = −1. 
1. Every time our model marked a point as **<span style="color: blue;">blue</span>** and this point lies in the range of the **<span style="color: green;">green</span>** area, our model predicted the correct answer. 
1. Every time our model marked a point as **<span style="color: purple;">purple</span>** and this point lies in the range of the **<span style="color: red;">red</span>** area, our model predicted the correct answer. 
1. Every time our model marked a point as **<span style="color: blue;">blue</span>** and this point lies in the range of the **<span style="color: red;">red</span>** area, our model predicted the wrong answer. 
1. Every time our model marked a point as **<span style="color: purple;">purple</span>** and this point lies in the range of the **<span style="color: green;">green</span>** area, our model predicted the wrong answer. 

As we can see, near the line where  **y = 0** the model had a few mistakes. 

Because our score is almost perfect and Back-Propagation is much more accurate than Adaline we won’t perform the Back-Propagation Algorithm on the part A function. 

In part B we scored  <span style="font-size:15.0pt"> **99.5%**</span> and we thought that this is too good to be true.
We did some digging and we found out that the range of the points that can get a value of **1** is small if we compare it to the range of all the points. This means a high probability of a point
(almost 100%) landing in the **−1** range which can cause our model to overfit and always predict  **−1**.  
As we can see in the confusion matrix below, we got amazing result but the model predict most of the time **−1**: 

![](readme_pics/im4.png)


So we decided to balance the data by giving our model approximately half instances from
each type.   
After doing so our model accuracy was quite bad <span style="font-size:15.0pt"> **55.87%**</span> with Adaline Algorithm. 
![](readme_pics/im5.png)

Finally, now we will try to improve our model accuracy by using the Back-Propagation Algorithm using the MLP. 

Let’s recall our problem: 

![](readme_pics/im2.png)

We solved this problem using the MLP Classifier from Sklearn.  
The MLPClassifier uses Feed-Forward Networks and Back-propagation, with a given data   
and desired output, the MLPClassifier train the model and changes the weight and bias accordingly. 

We chose to do it with 2 hidden layers: 

1. The first hidden layer will have 8 neurons. 
1. The second hidden layer will have 2 neurons. 

Below we can see a diagram for this network:

![](readme_pics/im6_1.jpg)
![](readme_pics/im6_2.jpg)

Because our formula is: 

![](readme_pics/im2.png)

It makes sense that all the green area will look like the space with: 

All the circles such that: **2 ≤ Radius ≤ 3**,  and with the center of (0,0), and with  **x,y** such that:  **{4 ≤ x^2 + Y^2 ≤ 9}**.
_____

To conclude the Back-Propagation Algorithm using the MLP result of part B with the balanced dataset was  <span style="font-size:15.0pt"> **98.28%**</span>.  
Which is almost twice as good as the Adaline score which was  <span style="font-size:15.0pt"> **55.87%**</span>.  
We can see from the Confusion Matrix that the model doesn’t overfit: 

*left matrix - Back-Propagation, right matrix - Adaline* 

![](readme_pics/Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.042.jpeg)![](readme_pics/Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.043.png)

##Part D:

In this part, we will use our trained neurons from the next to last level of part C as our input.   
The last neuron will do the prediction according to the Adaline algorithm and not the Back-Propagation  
Algorithm that we used in part C, and we will train only the **Adaline** neuron. 

Our problem is the same as before:  

![](readme_pics/im2.png)

After doing so, combining the MLP Classifier with the Adaline Algorithm, our model  score was  <span style="font-size:15.0pt"> **98.3%**</span>.

![](readme_pics/Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.044.png)

Which is almost the same as the result of the Back-Propagation Algorithm  <span style="font-size:15.0pt"> **98.28%**</span>.  
Below we can see a diagram for this network:

![](readme_pics/im7_1.jpg)
![](readme_pics/im7_2.jpg)

As we saw in the diagram the scatters of the Adaline using the MLP and the Back-Propagation using the MLP are   
almost identical, which is why both of the scores are almost the same.   

From the Confusion Matrix, we can conclude that both models’ prediction was quite similar. 

###Adaline with MLP 

![](readme_pics/Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.050.jpeg)

###Back-Propagation with MLP

![](readme_pics/Aspose.Words.a56d60de-3c55-4e00-a247-60954006dbd8.051.jpeg)

We can see that the only difference between those models is that Adaline with MLP predicted  2  points as −1 while Back-Propagation with MLP predicted apparently those points as 1. 

To conclude, our first Adaline (from part B) score was: <span style="font-size:15.0pt"> **55.879%**</span>,  
The Back-Propagation with MLP score was <span style="font-size:15.0pt"> **98.28%**</span>,  
and the Adaline with MLP score was <span style="font-size:15.0pt"> **98.3%**</span>. 

So we saw that MLP was able to improve our model and by using MLP in two different Algorithms we got almost the same scores. 
