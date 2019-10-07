# NER Project

### Spacy

spaCy is a library for advanced Natural Language Processing in Python and Cython. It's built on the very latest research, and was designed from day one to be used in real products. 

### Meaning of the project

- Named Entity Recognition  


- Get the accuracy of the model

the proportion of train data, dev data and test data is 8:1:1
<table>
  <thead>
    <tr>
      <th>train_data</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>f-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ave/total</td>
      <th>98.6</td>
      <th>99.7</th>
      <th>99.2</th>
      <th>787</th>
    </tr>
    <tr>
      <td>ORG</td>
      <th>100</th>
      <th>100</th>
      <th>100</th>
      <th>348</td>
    </tr>
    <tr>
      <td>FINANCIAL</td>
      <th>97.8</th>
      <th>100</th>
      <th>98.9</th>
      <th>272</td>
    </tr>
    <tr>
      <td>CONCEPT</td>
      <th>96.1</th>
      <th>99.2</th>
      <th>97.6</th>
      <th>129</td>
    </tr>
    <tr>
      <td>LOC </td>
      <th>100</th>
      <th>100</th>
      <th>100</th>
      <th>4</td>
    </tr>
    <tr>
      <td>PER</td>
      <th>100</th>
      <th>96</th>
      <th>97.9</th>
      <th>25</td>
    </tr>
    <tr>
      <td>MISC</td>
      <th>100</th>
      <th>100</th>
      <th>100</th>
      <th>9</td>
    </tr>
  </tbody>
</table>




<table>
  <thead>
    <tr>
      <th>test_data</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>f-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ave/total</td>
      <th>66.7</td>
      <th>68.6</th>
      <th>67.6</th>
      <th>137</th>
    </tr>
    <tr>
      <td>ORG</td>
      <th>72</th>
      <th>83.7</th>
      <th>77.4</th>
      <th>57</td>
    </tr>
    <tr>
      <td>FINANCIAL</td>
      <th>61.8</th>
      <th>67,7</th>
      <th>64.6</th>
      <th>44</td>
    </tr>
    <tr>
      <td>CONCEPT</td>
      <th>62.5</th>
      <th>58.8</th>
      <th>60.6</th>
      <th>23</td>
    </tr>
    <tr>
      <td>LOC </td>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>3</td>
    </tr>
    <tr>
      <td>PER</td>
      <th>60</th>
      <th>50</th>
      <th>54.5</th>
      <th>8</td>
    </tr>
    <tr>
      <td>MISC</td>
      <th>0</th>
      <th>0</th>
      <th>0</th>
      <th>2</td>
    </tr>
  </tbody>
</table>