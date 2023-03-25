# 01- Öznitelik Seçimi ve Öznitelik Mühendisliği

**1. Adım: Veri Ön İşleme**

Bu aşamada veri işlenmeden önce daha temiz bir veri seti oluşturmak için yapılacak olan işlemlerin uygulaması yapılmıştır.

-Kullanılan veri setinde "esya" özniteliği tüm veriler için aynı değeri almıştır. Bu öznitelik sonucu etkilemeyeceği için kaldırılmıştır.

-"siteAdi" ve "aidat" özniteliği veri setinin çoğunda belirtilmemiştir. Sonuca etkisi olmadığı için bu özniteliklerde kaldırılmıştır.
```
del evVeriSeti['esya']
del evVeriSeti['siteAdi']
del evVeriSeti['aidat'] 
```

-`evVeriSeti.dropna()` ile veri setinde NaN değer alan satırlar silinmiştir.

**ExtraTreesClassifier ile Öznitelik Önemi**

Öznitelik önemi bir veri seti içerisindeki en yararlı öznitelikleri bulmak ve karşılaştırmak için kullanılır.

```
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as pltdata = evVeriSeti.dropna()
X = data.iloc("fiyat", axis=1) #bağımsız değişkenler
y = data.iloc["fiyat"] #bağımlı değişken
model = ExtraTreesClassifier()
model.fit(X,y) 
print(model.feature_importances_) #burada özniteliklerimizin önemi sırayla ekrana yazdırılımıştır

#grafik olarak gösterelim
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

```

**Veri Setindeki En İyi Özniteliklerin Skorlandırılması (SelectKBest)**

Makine Öğrenmesi çalışmlarında başarılı bir tahmin için, doğru öznitelik seçimi önemlidir. Yapılan çalışmada bu aşamada Chi-square yöntemi kullanılmıştır. Chi-square özniteliklerin bağımlı değişken ile arasındaki ilişkiyi skorlandırır. Skor değeri azaldıkça öznitelik bağımsızlığı artmaktadır.

```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
scores = pd.concat([dfcolumns,dfscores],axis=1)
scores.columns = ['özellikler','score']
print(scores.nlargest(5,'score'))
```

**Korelasyon Isı Haritası**
```
import seaborn as sns
corr = evVeriSeti.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values
```

**2. Adım: Ölçeklendirme**
Veri setinde bulunan değerlerin ölçek farkı makine öğrenmesi çalışmalarında yanlış sonuçlar almamıza sebep olabilir. Bu durumu ortadan kaldırmak için ölçeklendirme yapılmaktadır.

**Standartlaştırma**
```
from sklearn.preprocessing import StandardScaler
stdscaler=StandartScaler()
evVeriSeti_stdscaled=pd.DataFrame(stdscaler.fit_transform(evVeriSeti.dropna()),columns=evVeriSeti
```
**Normalizasyon**

Veri setindeki veriler 0 ile 1  arasında değerlere atanır. bu sayde veriler arasındaki benzerlik arttırılmış olup, farklı özelliklerin birbirleri ile karşılaştırılması daha kolay hale getirilmiştir.

```
from sklearn.preprocessing import MinMaaxScaler
mmsscaler=StandartScaler()
evVeriSeti_mmsscaled=pd.DataFrame(mmsscaler.fit_transform(evVeriSeti.dropna()),columns=evVeriSeti
```
**3. Adım: Temel İstatistik**
Verilerin genel özelliklerini anlamk için temel istatiklerini hesaplamak gerekmektedir.

`evVeriSeti.describe()` metodu ile basit bir şekilde hesaplanabilir. Bu metot ile aşağıdaki veriler değerler hesaplanmaktadır.

count:Girdi sayısı

mean(aritmetik ortalama):Özellik sütunun ortalaması

std(standart sapma):Özellik sütunun  standart sapması

min:Özelliğin aldığı minimum değer

max:Özelliğin aldığı maximum değer

50%:Özelliğin ortadaki değeri

25%:Medyan ile minimum arasındaki değer

75%:Medyan ile maximum değer arasındaki değer

**Mod**

Verideki en çok tekrar eden sayıyı ifade etmektedir. `evVeriSeti_mod=evVeriSeti.mode()` metodu ile hesaplanır.

**Medyan**

Verideki özellikler sıralandığında (ayrı ayrı) ortada kalan sayıyı ifade etmektedir. `evVeriSeti.median()` metodu ile hesaplanır.

**Kovaryans**

İki değişkenin birlikte ne kadar değiştiklerinin ölçüsüdür.` evVeriSeti.cov()` metodu ile hesaplanır.



