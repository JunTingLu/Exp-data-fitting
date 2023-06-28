# Exp-data-fitting 

由於在光纖/矽晶片耦合實驗中，時常需要量測模場直徑(MFD)來確認傳播媒介的"敏感度"，以藉此了解傳播的損耗程度，
根據實驗上量測數據，由於環境設備微擾使得實際得到數據有許多浮動，因此利用機器學習的方法，並利用資料搜尋演算
法找出預測之數據曲線中最接近MFD的數值<h1>
# 專案相關工具
* Keras 框架
* python
# 檔案資料夾說明範例
  <table>
   <tr>
    <td>main.py</td>
    <td>主執行檔，進行模型預測及最佳值計算</td>
  </tr>
    <tr>
    <td>export.py</td>
    <td>匯出main.py所得到之數據並最終以excel格式產出</td>
  </tr>
     <tr>
    <td>info.ini</td>
    <td>紀錄main.py所計算之數據</td>
  </tr>
  </table>
  
# 開發環境
 Visual Stdio 

 
