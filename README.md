<h1> Experimental Data Fitting </h1>
  
由於在光纖/矽晶片耦合實驗中，時常需要量測模場直徑(MFD)來確認傳播媒介的"敏感度"，以藉此了解傳播的損耗程度，
根據實驗上量測數據，在環境設備微擾下使得實際得到數據有許多浮動，因此利用機器學習的方法，並利用資料搜尋演算
法找出預測之數據曲線中最接近MFD的數值

<h2>前置作業</h2> 

> 確認檔案為.xlsx (否則須轉檔)

<h2>檔案資料夾說明範例</h2> 
  <table>
   <tr>
    <td>main.py</td><td>主執行檔，進行模型預測及最佳值計算</td>
  </tr>
    <tr>
    <td>export.py</td><td>匯出main.py所得到之數據並最終以excel格式產出</td>
  </tr>
     <tr>
    <td>info.ini</td><td>紀錄main.py所計算之數據</td>
  </tr>
     <tr>
    <td>imgs</td><td>儲存擬合曲線</td>
  </tr>
  </table>
  
<h2>程式語言和套件</h2> 

 - Python
 - Keras

 
