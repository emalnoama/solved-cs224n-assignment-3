Download Link: https://assignmentchef.com/product/solved-cs224n-assignment-3
<br>
<strong>Note: In this assignment, the inputs to neural network layers will be row vectors because this is standard practice for TensorFlow (some built-in TensorFlow functions assume the inputs are row vectors). This means the weight matrix of a hidden layer will right-multiply instead of left-multiply its input (i.e., xW </strong>+ <strong>b instead of Wx </strong>+ <strong>b).</strong>

<h1>A primer on named entity recognition</h1>

In this assignment, we will build several different models for named entity recognition (NER). NER is a subtask of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. In the assignment, for a given a word in a context, we want to predict whether it represents one of four categories:

<ul>

 <li>Person (PER): e.g. “Martha Stewart”, “Obama”, “Tim Wagner”, etc. Pronouns like “he” or “she” are <em>not </em>considered named entities.</li>

 <li>Organization (ORG): e.g. “American Airlines”, “Goldman Sachs”, “Department of Defense”.</li>

 <li>Location (LOC): e.g. “Germany”, “Panama Strait”, “Brussels”, but not unnamed locations like “the bar” or “the farm”.</li>

 <li>Miscellaneous (MISC): e.g. “Japanese”, “USD”, “1,000”, “Englishmen”.</li>

</ul>

We formulate this as a 5-class classification problem, using the four above classes and a null-class (O) for words that do not represent a named entity (most words fall into this category). For an entity that spans multiple words (“Department of Defense”), each word is separately tagged, and every contiguous sequence of non-null tags is considered to be an entity.

Here is a sample sentence (<strong>x</strong><sup>(<em>t</em>)</sup>) with the named entities tagged above each token (<strong>y</strong><sup>(<em>t</em>)</sup>) as well as hypothetical predictions produced by a system (<strong>y</strong>ˆ<sup>(<em>t</em>)</sup>):

1

<table width="568">

 <tbody>

  <tr>

   <td width="34"><strong>y</strong>(<em>t</em>)</td>

   <td width="69"><strong>ORG</strong></td>

   <td width="62"><strong>ORG</strong></td>

   <td width="57"><strong>O O</strong></td>

   <td width="24"><strong>O</strong></td>

   <td width="45"><strong>ORG</strong></td>

   <td width="49"><strong>ORG</strong></td>

   <td width="31">…</td>

   <td width="76"><strong>O</strong></td>

   <td width="95"><strong>PER PER</strong></td>

   <td width="27"><strong>O</strong></td>

  </tr>

  <tr>

   <td width="34"><strong>y</strong>ˆ(<em>t</em>)</td>

   <td width="69">MISC</td>

   <td width="62">O</td>

   <td width="57">O O</td>

   <td width="24">O</td>

   <td width="45">ORG</td>

   <td width="49">O</td>

   <td width="31">…</td>

   <td width="76">O</td>

   <td width="95">PER PER</td>

   <td width="27">O</td>

  </tr>

  <tr>

   <td width="34"><strong>x</strong>(<em>t</em>)</td>

   <td width="69">American</td>

   <td width="62">Airlines,</td>

   <td width="57">a     unit</td>

   <td width="24">of</td>

   <td width="45">AMR</td>

   <td width="49">Corp.,</td>

   <td width="31">…</td>

   <td width="76">spokesman</td>

   <td width="95">Tim     Wagner</td>

   <td width="27">said.</td>

  </tr>

 </tbody>

</table>

In the above example, the system mistakenly predicted “American” to be of the MISC class and ignores “Airlines” and “Corp.”. All together, it predicts 3 entities, “American”, “AMR” and “Tim Wagner”.

To evaluate the quality of a NER system’s output, we look at precision, recall and the <em>F</em><sub>1 </sub>measure.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> In particular, we will report precision, recall and <em>F</em><sub>1 </sub>at both the token-level and the name-entity level. In the former case:

<ul>

 <li>Precision is calculated as the ratio of correct non-null labels predicted to the total number of non-null labels predicted (in the above example, it would be</li>

 <li>Recall is calculated as the ratio of correct non-null labels predicted to the total number of <em>correct </em>non-null labels (in the above example, it would be</li>

 <li><em>F</em><sub>1 </sub>is the harmonic mean of the two: . (in the above example, it would be</li>

</ul>

For entity-level <em>F</em><sub>1</sub>:

<ul>

 <li>Precision is the fraction of predicted entity name spans that line up exactly with spans in the gold standard evaluation data. In our example, “AMR” would be marked incorrectly because it does not cover the whole entity, i.e. “AMR Corp.”, as would “American”, and we would get a precision score of</li>

</ul>

.

<ul>

 <li>Recall is similarly the number of names in the gold standard that appear at exactly the same location in the predictions. Here, we would get a recall score of .</li>

 <li>Finally, the <em>F</em><sub>1 </sub>score is still the harmonic mean of the two, and would be in the example.</li>

</ul>

Our model also outputs a token-level <em>confusion matrix</em><a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>. A confusion matrix is a specific table layout that allows visualization of the classification performance. Each column of the matrix represents the instances in a predicted class while each row represents the instances in an actual class. The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabelling one as another).

<h1>1.     A window into NER</h1>

Let’s look at a simple baseline model that predicts a label for each token separately using features from a window around it.

Figure 1:     A sample input sequence

def

Figure 1 shows an example of an input sequence and the first window from this sequence. Let <strong>x </strong>= <strong>x</strong><sup>(1)</sup><em>,</em><strong>x</strong><sup>(2)</sup><em>,…,</em><strong>x</strong><sup>(<em>T</em>) </sup>be an input sequence of length <em>T </em>and <strong>y </strong><sup>def</sup>= <strong>y</strong><sup>(1)</sup><em>,</em><strong>y</strong><sup>(2)</sup><em>,…,</em><strong>y</strong><sup>(<em>T</em>) </sup>be an output sequence, also of length <em>T</em>. Here, each element <strong>x</strong><sup>(<em>t</em>) </sup>and <strong>y</strong><sup>(<em>t</em>) </sup>are one-hot vectors representing the word at the t-th index of the sentence. In a window based classifier, every input sequence is split into <em>T </em>new data points, each representing a window and its label. A new input is constructed from a window around <strong>x</strong><sup>(<em>t</em>) </sup>by concatenating <em>w </em>tokens to the left and right of <strong>x</strong><sup>(<em>t</em>)</sup>: <strong>x˜</strong><sup>(<em>t</em>) def</sup>= [<strong>x</strong><sup>(<em>t</em>−<em>w</em>)</sup><em>,…,</em><strong>x</strong><sup>(<em>t</em>)</sup><em>,…,</em><strong>x</strong><sup>(<em>t</em>+<em>w</em>)</sup>]; we continue to use <strong>y</strong><sup>(<em>t</em>) </sup>as its label. For windows centered around tokens at the very beginning of a sentence, we add special start tokens (&lt;START&gt;) to the beginning of the window and for windows centered around tokens at the very end of a sentence, we add special end tokens (&lt;END&gt;) to the end of the window. For example, consider constructing a window around “Jim” in the sentence above. If window size were 1, we would add a <em>single </em>start token to the window (resulting in a window of [&lt;START&gt;, Jim, bought]). If window size were 2, we would add <em>two </em>start tokens to the window (resulting in a window of [&lt;START&gt;, &lt;START&gt;, Jim, bought, 300]).

With these, each input and output is of a uniform length (<em>w </em>and 1 respectively) and we can use a simple feedforward neural net to predict <strong>y</strong><sup>(<em>t</em>) </sup>from <strong>x˜</strong><sup>(<em>t</em>)</sup>:

As a simple but effective model to predict labels from each window, we will use a single hidden layer with a ReLU activation, combined with a softmax output layer and the cross-entropy loss:

<strong>e</strong>(<em>t</em>) = [<strong>x</strong>(<em>t</em>−<em>w</em>)<em>L,…,</em><strong>x</strong>(<em>t</em>)<em>L,…,</em><strong>x</strong>(<em>t</em>+<em>w</em>)<em>L</em>] <strong>h</strong>(<em>t</em>) = ReLU(<strong>e</strong>(<em>t</em>)<em>W </em>+ <strong>b</strong>1)

<strong>y</strong>ˆ<sup>(<em>t</em>) </sup>= softmax(<strong>h</strong><sup>(<em>t</em>)</sup><em>U </em>+ <strong>b</strong><sub>2</sub>)

<em>J </em>= CE(<strong>y</strong>(<em>t</em>)<em>,</em><strong>y</strong>ˆ(<em>t</em>))

CE(<strong>y</strong>(<em>t</em>)<em>,</em><strong>y</strong>ˆ(<em>t</em>)) = −X<em>y</em><em>i</em>(<em>t</em>) log(ˆ<em>y</em><em>i</em>(<em>t</em>))<em>.</em>

<em>i</em>

where <em>L </em>∈R<em><sup>V</sup></em><sup>×<em>D </em></sup>are word embeddings, <strong>h</strong><sup>(<em>t</em>) </sup>is dimension <em>H </em>and <strong>y</strong>ˆ<sup>(<em>t</em>) </sup>is of dimension <em>C</em>, where <em>V </em>is the size of the vocabulary, <em>D </em>is the size of the word embedding, <em>H </em>is the size of the hidden layer and <em>C </em>are the number of classes being predicted (here 5).

<ul>

 <li> (written)

  <ol>

   <li> Provide 2 examples of sentences containing a named entity with an ambiguous type (e.g. the entity could either be a person or an organization, or it could either be an organization or not an entity).</li>

   <li> Why might it be important to use features apart from the word itself to predict named entity labels?</li>

  </ol></li>

</ul>

<ul>

 <li> Describe at least two features (apart from the word) that would help in predicting whether a word is part of a named entity or not.</li>

</ul>

<ul>

 <li> (written)

  <ol>

   <li> What are the dimensions of <strong>e</strong><sup>(<em>t</em>)</sup>, <em>W </em>and <em>U </em>if we use a window of size <em>w</em>?</li>

   <li> What is the computational complexity of predicting labels for a sentence of length <em>T</em>?</li>

  </ol></li>

 <li> (code) Implement a window-based classifier model in py using this approach.</li>

</ul>

To do so, you will have to:

<ol>

 <li> Transform a batch of input sequences into a batch of windowed input-output pairs in the makewindoweddata You can test your implementation by running python q1window.py test1.</li>

 <li>Implement the feed-forward model described above by appropriately completing functions in the WindowModel You can test your implementation by running python q1window.py test2.</li>

</ol>

<ul>

 <li> Train your model using the command python q1window.py train. The code should take only about 2–3 minutes to run and you should get a development score of at least 81% <em>F</em><sub>1</sub>.</li>

</ul>

The model and its output will be saved to results/window/&lt;timestamp&gt;/, where &lt;timestamp&gt; is the date and time at which the program was run. The file results.txt contains formatted output of the model’s predictions on the development set, and the file log contains the printed output, i.e. confusion matrices and <em>F</em><sub>1 </sub>scores computed during the training.

Finally, you can interact with your model using:

<h2>python q1window.py shell -m results/window/&lt;timestamp&gt;/</h2>

<strong>Deliverable: </strong>After your model has trained, copy the windowpredictions.conll file from the appropriate results folder into the root folder of your code directory, so that it can be included in your submission.

(d)  (written) Analyze the predictions of your model using the files generated above.

<ol>

 <li> Report your best development entity-level <em>F</em><sub>1 </sub>score and the corresponding token-level confusion matrix. Briefly describe what the confusion matrix tells you about the errors your model is making.</li>

 <li> Describe at least 2 modeling limitations of the window-based model and support these conclusions using examples from your model’s output (i.e. identify errors that your model made due to its limitations). You can also support your conclusions using predictions made by your model on examples manually entered through the shell.</li>

</ol>

<h1>2.     Recurrent neural nets for NER</h1>

We will now tackle the task of NER by using a recurrent neural network (RNN).

Recall that each RNN cell combines the hidden state vector with the input using a sigmoid. We then

use the hidden state to predict the output at each timestep:

<strong>e</strong>(<em>t</em>) = <strong>x</strong>(<em>t</em>)<em>L </em><strong>h</strong>(<em>t</em>) = <em>σ</em>(<strong>h</strong>(<em>t</em>−1)<em>W</em><em>h </em>+ <strong>e</strong>(<em>t</em>)<em>W</em><em>x </em>+ <strong>b</strong>1)

<strong>y</strong>ˆ<sup>(<em>t</em>) </sup>= softmax(<strong>h</strong><sup>(<em>t</em>)</sup><em>U </em>+ <strong>b</strong><sub>2</sub>)<em>,</em>

where <em>L </em>∈R<em><sup>V</sup></em><sup>×<em>D </em></sup>are word embeddings, <em>W<sub>h </sub></em>∈R<em><sup>H</sup></em><sup>×<em>H</em></sup>, <em>W<sub>x </sub></em>∈R<em><sup>D</sup></em><sup>×<em>H </em></sup>and <strong>b</strong><sub>1 </sub>∈R<em><sup>H </sup></em>are parameters for the RNN cell, and <em>U </em>∈R<em><sup>H</sup></em><sup>×<em>C </em></sup>and <strong>b</strong><sub>2 </sub>∈R<em><sup>C </sup></em>are parameters for the softmax. As before, <em>V </em>is the size of the vocabulary, <em>D </em>is the size of the word embedding, <em>H </em>is the size of the hidden layer and <em>C </em>are the number of classes being predicted (here 5).

In order to train the model, we use a cross-entropy loss for the every predicted token:

<em>T</em>

<em>J </em>= X<em>CE</em>(<strong>y</strong>(<em>t</em>)<em>,</em><strong>y</strong>ˆ(<em>t</em>))

<em>t</em>=1

<em>CE</em>(<strong>y</strong>(<em>t</em>)<em>,</em><strong>y</strong>ˆ(<em>t</em>)) = −X<em>y</em><em>i</em>(<em>t</em>) log(ˆ<em>y</em><em>i</em>(<em>t</em>))<em>.</em>

<em>i</em>

<ul>

 <li> (written)

  <ol>

   <li> How many more parameters does the RNN model in comparison to the window-based model?</li>

   <li> What is the computational complexity of predicting labels for a sentence of length <em>T </em>(for the RNN model)?</li>

  </ol></li>

 <li> (written) Recall that the actual score we want to optimize is entity-level <em>F</em><sub>1</sub>.

  <ol>

   <li> Name at least one scenario in which decreasing the cross-entropy cost would lead to an <em>decrease </em>in entity-level <em>F</em><sub>1 </sub></li>

   <li> Why it is difficult to directly optimize for <em>F</em><sub>1</sub>?</li>

  </ol></li>

 <li>(code) Implement an RNN cell using the equations described above in the rnncell function of py. You can test your implementation by running python q2rnncell.py test.</li>

 <li> (code/written) Implementing an RNN requires us to unroll the computation over the whole sentence. Unfortunately, each sentence can be of arbitrary length and this would cause the RNN to be unrolled a different number of times for different sentences, making it impossible to batch process the data.</li>

</ul>

The most common way to address this problem is <em>pad </em>our input with zeros. Suppose the largest sentence in our input is <em>M </em>tokens long, then, for an input of length <em>T </em>we will need to:

<ol>

 <li>Add 0-vectors to <strong>x </strong>and <strong>y </strong>to make them <em>M </em>tokens long.</li>

 <li>Create a <em>masking vector</em>, ( which is 1 for all <em>t </em>≤ <em>T </em>and 0 for all <em>t &gt; T</em>. This masking vector will allow us to ignore the predictions that the network makes on the padded input.<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a></li>

 <li>Of course, by extending the input and output by <em>M </em>− <em>T </em>tokens, we might change our loss and hence gradient updates. In order to tackle this problem, we modify our loss using the masking vector:</li>

</ol>

<em>M</em>

<em>J </em>= X<em>m</em>(<em>t</em>) CE(<em>y</em>(<em>t</em>)<em>,y</em>ˆ(<em>t</em>))<em>.</em>

<em>t</em>=1

<ol>

 <li>(3 points) (written) How would the loss and gradient updates change if we did not use masking?</li>

</ol>

How does masking solve this problem?

<ol>

 <li>(5 points) (code) Implement padsequences in your code. You can test your implementation by running python q2rnn.py test1.</li>

</ol>

<ul>

 <li>(12 points) (code) Implement the rest of the RNN model assuming only fixed length input by appropriately completing functions in the RNNModel This will involve:

  <ol>

   <li>Implementing the addplaceholders, addembedding, addtrainingop</li>

   <li>Implementing the addpredictionop operation that unrolls the RNN loop maxlength times. Remember to <em>reuse </em>variables in your variable scope from the 2nd timestep onwards to share the RNN cell weights <em>W<sub>x </sub></em>and <em>W<sub>h </sub></em>across timesteps.</li>

   <li>Implementing addlossop to handle the mask vector returned in the previous part.</li>

  </ol></li>

</ul>

You can test your implementation by running python q2rnn.py test2.

<ul>

 <li>(3 points) (code) Train your model using the command python q2rnn.py train. Training should take about 2 hours on your CPU and 10–20 minutes if you use the GPUs provided by Microsoft Azure. You should get a development <em>F</em><sub>1 </sub>score of at least 85%.</li>

</ul>

The model and its output will be saved to results/rnn/&lt;timestamp&gt;/, where &lt;timestamp&gt; is the date and time at which the program was run. The file results.txt contains formatted output of the model’s predictions on the development set, and the file log contains the printed output, i.e. confusion matrices and <em>F</em><sub>1 </sub>scores computed during the training.

Finally, you can interact with your model using:

<h2>python q2rnn.py shell -m results/rnn/&lt;timestamp&gt;/</h2>

<strong>Deliverable: </strong>After your model has trained, copy the rnnpredictions.conll file from the appropriate results folder into your code directory so that it can be included in your submission.

(g) (6 points) (written)

<ol>

 <li>(3 points) Describe at least 2 modeling limitations of this RNN model and support these conclusions using examples from your model’s output.</li>

 <li>(3 points) For each limitation, suggest some way you could extend the model to overcome the limitation.</li>

</ol>

<h1>3.     Grooving with GRUs</h1>

In class, we learned that a gated recurrent unit (GRU) is an improved RNN cell that greatly reduces the problem of vanishing gradients. Recall that a GRU is described by the following equations:

<strong>z</strong>(<em>t</em>) = <em>σ</em>(<strong>x</strong>(<em>t</em>)<em>U</em><em>z </em>+ <strong>h</strong>(<em>t</em>−1)<em>W</em><em>z </em>+ <strong>b</strong><em>z</em>) <strong>r</strong>(<em>t</em>) = <em>σ</em>(<strong>x</strong>(<em>t</em>)<em>U</em><em>r </em>+ <strong>h</strong>(<em>t</em>−1)<em>W</em><em>r </em>+ <strong>b</strong><em>r</em>) <strong>h˜</strong>(<em>t</em>) = tanh(<strong>x</strong>(<em>t</em>)<em>U</em><em>h </em>+ <strong>r</strong>(<em>t</em>) ◦<strong>h</strong>(<em>t</em>−1)<em>W</em><em>h </em>+ <strong>b</strong><em>h</em>) <strong>h</strong>(<em>t</em>) = <strong>z</strong>(<em>t</em>) ◦<strong>h</strong>(<em>t</em>−1) + (1 −<strong>z</strong>(<em>t</em>)) ◦<strong>h˜</strong>(<em>t</em>)<em>,</em>

where <strong>z</strong><sup>(<em>t</em>) </sup>is considered to be an <em>update gate </em>and <strong>r</strong><sup>(<em>t</em>) </sup>is considered to be a <em>reset gate</em>.<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>

Also, to keep the notation consistent with the GRU, for this problem, let the basic RNN cell be described by the equations:

<strong>h</strong>(<em>t</em>) = <em>σ</em>(<strong>x</strong>(<em>t</em>)<em>U</em><em>h </em>+ <strong>h</strong>(<em>t</em>−1)<em>W</em><em>h </em>+ <strong>b</strong><em>h</em>)<em>.</em>

To gain some inutition, let’s explore the behavior of the basic RNN cell and the GRU on some generated 1-D sequences.

<ul>

 <li>(4 points) (written) <strong>Modeling latching behavior. </strong>Let’s say we are given input sequences starting with a 1 or 0, followed by <em>n </em>0s, e.g. 0, 1, 00, 10, 000, 100, etc. We would like our state <em>h </em>to continue to remember what the first character was, irrespective of how many 0s follow. This scenario can also be described as wanting the neural network to learn the following simple automaton:</li>

</ul>

<em>x </em>= 0<em>,</em>1

In other words, when the network sees a 1, it should change its state to also be a 1 and stay there.

In the following questions, assume that the state is initialized at 0 (i.e. <em>h</em><sup>(0) </sup>= 0), and that all the parameters are scalars. Further, assume that all sigmoid activations and tanh activations are replaced by the indicator function:

(                                                                                (

1          if <em>x &gt; </em>0    1              if <em>x &gt; </em>0 <em>σ</em>(<em>x</em>) →      tanh(<em>x</em>) →             <em>.</em>

0    otherwise                                                         0    otherwise

<ol>

 <li>(1 point) Identify values of <em>w<sub>h</sub></em>, <em>u<sub>h </sub></em>and <em>b<sub>h </sub></em>for an RNN cell that would allow it to replicate the behavior described by the automaton above.</li>

 <li>(3 points) Let <em>w<sub>r </sub></em>= <em>u<sub>r </sub></em>= <em>b<sub>r </sub></em>= <em>b<sub>z </sub></em>= <em>b<sub>h </sub></em>= 0. Identify values of <em>w<sub>z</sub></em>, <em>u<sub>z</sub></em>, <em>w<sub>h </sub></em>and <em>u<sub>h </sub></em>for a GRU cell that would allow it to replicate the behavior described by the automaton above.</li>

</ol>

<ul>

 <li>(6 points) (written) <strong>Modeling toggling behavior. </strong>Now, let us try modeling a more interesting behavior. We are now given an arbitrary input sequence, and must produce an output sequence that switches from 0 to 1 and vice versa whenever it sees a 1 in the input. For example, the input sequence 00100100 should produce 00111000. This behavior could be described by the following automaton:</li>

</ul>

<em>x </em>= 1

Once again, assume that the state is initialized at 0 (i.e. <em>h</em><sup>(0) </sup>= 0), that all the parameters are scalars, that all sigmoid activations and tanh activations are replaced by the indicator function.

<ol>

 <li>(3 points) Show that a 1D RNN can not replicate the behavior described by the automaton above.</li>

 <li>(3 points) Let <em>w<sub>r </sub></em>= <em>u<sub>r </sub></em>= <em>b<sub>z </sub></em>= <em>b<sub>h </sub></em>= 0. Identify values of <em>b<sub>r</sub></em>, <em>w<sub>z</sub></em>, <em>u<sub>z</sub></em>, <em>w<sub>h </sub></em>and <em>u<sub>h </sub></em>for a GRU cell that would allow it to replicate the behavior described by the automaton above.</li>

</ol>

<ul>

 <li>(6 points) (code) Implement the GRU cell described above in py. You can test your implementation by running python q3grucell.py test.</li>

 <li>(6 points) (code) We will now use an RNN model to try and learn the latching behavior described in part (a) using TensorFlow’s RNN implementation: nn.dynamicrnn.

  <ol>

   <li>In py, implement addpredictionop by applying TensorFlow’s dynamic RNN model on the sequence input provided. Also apply a sigmoid function on the final state to normalize the state values between 0 and 1.</li>

   <li>Next, write code to calculate the gradient norm and implement gradient clipping in addtrainingop.</li>

  </ol></li>

</ul>

<ul>

 <li>Run the program:</li>

</ul>

<h2>python q3gru.py predict -c [rnn|gru] [-g]</h2>

to generate a learning curve for this task for the RNN and GRU models. The -g flag activates gradient clipping.

These commands produce a plot of the learning dynamics in q3-noclip-&lt;model&gt;.png and q3-clip-&lt;model&gt;.png respectively.

<strong>Deliverable: </strong>Attach the plots of learning dynamics generated for a GRU and RNN, with and without gradient clipping (in total 4 plots) to your write up.

<ul>

 <li>(5 points) (written) Analyze the graphs obtained above and describe the learning dynamics you see. Make sure you address the following questions:

  <ol>

   <li>Does either model experience vanishing or exploding gradients? If so, does gradient clipping help?</li>

   <li>Which model does better? Can you explain why?</li>

  </ol></li>

 <li>(3 points) (code) Run the NER model from question 2 using the GRU cell using the command: python q2rnn.py train -c gru</li>

</ul>

Training should take about 3–4 hours on your CPU and about 30 minutes if you use the GPUs provided by Microsoft Azure. You should get a development <em>F</em><sub>1 </sub>score of at least 85%.

The model and its output will be saved to results/gru/&lt;timestamp&gt;/, where &lt;timestamp&gt; is the date and time at which the program was run. The file results.txt contains formatted output of the model’s predictions on the development set, and the file log contains the printed output, i.e. confusion matrices and <em>F</em><sub>1 </sub>scores computed during the training.

Finally, you can interact with your model using:

<h2>python q2rnn.py shell -m results/gru/&lt;timestamp&gt;/</h2>

<strong>Deliverable: </strong>After your model has trained, copy the grupredictions.conll file from the appropriate results folder into your code directory so that it can be included in your submission.

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://en.wikipedia.org/wiki/Precision_and_recall">https://en.wikipedia.org/wiki/Precision_and_recall</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="https://en.wikipedia.org/wiki/Confusion_matrix">https://en.wikipedia.org/wiki/Confusion_matrix</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> In our code, we will actually use the Boolean values of True and False instead of 1 and 0 for computational efficiency.

<a href="#_ftnref4" name="_ftn4">[4]</a> Section 4.2.2 of <a href="https://arxiv.org/pdf/1511.07916.pdf">https://arxiv.org/pdf/1511.07916.pdf</a> provides a good introduction to GRUs. <a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">http://colah. </a><a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">github.io/posts/2015-08-Understanding-LSTMs/</a> provides a more colorful picture of LSTMs and to an extent GRUs.