# 房屋租赁—数据科学方法第 1 部分:使用 python 和 BeautifulSoup 收集所有信息—更新

> 原文：<https://medium.com/analytics-vidhya/house-rental-the-data-science-way-part-1-scrape-it-all-with-python-and-beautifulsoup-94d9d1222e32?source=collection_archive---------10----------------------->

![](img/da891c87a60b5de971dd2c1537410ef7.png)

去年我从原来的房子搬到了一个新的城市，换了工作。一切都进展得如此之快，在开始新工作之前，我只有几周的时间找到住处。在这种匆忙中，我没有足够的时间去了解这个城市的房地产市场，最终选择了在工作距离和服务之间取得更好平衡的住宿。但是……嗯……房间很小，我想我为那栋房子付了太多钱。但我只是猜测！

那…如果我能找出来呢？
那么……让我们回到我所学的机器学习。

# 构建您自己的数据集

这是机器学习应用的一个非常常见和标准的例子:对房价进行回归，以找出房价的真实成本。你可以在网上任何你想要的地方找到它的例子。除了他们通常使用来自…嗯…任何地方的数据集。我需要来自**的**新鲜价格**，我想要的城市**，我希望它可以在几个月内更新。只有一个办法:刮！

我住在意大利，确切地说是都灵。在意大利，收集所有租房或买房信息的最大网站是 www.immobiliare.it

Immobiliare .它收集了意大利每个机构可以用来展示他们所管理的建筑的公告，所以这可能是了解特定城市房地产市场的最佳方式。现在是我们动手的时候了。

# 做汤—更新

*更新:在夏季 immobiliare.it 的开发者用新的布局和更难浏览的 html 更新了网站。我将更新发布的代码以反映变化。*

我们要做的是上网站，导航到我们城市的主页，收集该城市所有地区的列表，并收集该地区发布的每个公告。
工具集:

```
**import** **requests**
**from** **bs4** **import** BeautifulSoup
**import** **pandas** **as** **pd**
**from** **tqdm** **import** tqdm_notebook **as** tqdm
**import** **csv**
```

现在我们准备好出发了！

```
def get_pages(main):
    try:
        soup = connect(main)
        n_pages = [_.get_text(strip=True) for _ in soup.find('ul', {'class': 'pagination pagination__number'}).find_all('li')]
        last_page = int(n_pages[-1])
        pages = [main]

        for n in range(2,last_page+1):    
            page_num = "/?pag={}".format(n)
            pages.append(main + page_num)
    except:
        pages = [main]

    return pagesdef connect(web_addr):
    resp = requests.get(web_addr)
    return BeautifulSoup(resp.content, "html.parser")def get_areas(website):
    data = connect(website)
    areas = []
    for ultag in data.find_all('ul', {'class': 'breadcrumb-list breadcrumb-list_list breadcrumb-list__related'}):
        for litag in ultag.find_all('li'):
            for i in range(len(litag.text.split(','))):
                areas.append(litag.text.split(',')[i])
    areas = [x.strip() for x in areas]
    urls = []

    for area in areas:
        url = website + '/' + area.replace(' ','-').lower()
        urls.append(url)

    return areas, urlsdef get_apartment_links(website):
    data = connect(website)
    links = []
    for link in data.find_all('ul', {'class': 'annunci-list'}):
        for litag in link.find_all('li'):
            try:
                links.append(litag.a.get('href'))
            except:
                continue
    return linksdef scrape_link(website):
    data = connect(website)
    info = data.find_all('dl', {'class': 'im-features__list'})
    comp_info = pd.DataFrame()
    cleaned_id_text = []
    cleaned_id__attrb_text = []
    for n in range(len(info)):
        for i in info[n].find_all('dt'):
            cleaned_id_text.append(i.text)
        for i in info[n].find_all('dd'):
            cleaned_id__attrb_text.append(i.text)comp_info['Id'] = cleaned_id_text
    comp_info['Attribute'] = cleaned_id__attrb_text
    comp_info
    feature = []
    for item in comp_info['Attribute']:
        try:
            feature.append(clear_df(item))
        except:
            feature.append(ultra_clear_df(item))comp_info['Attribute'] = feature
    return comp_info['Id'].values, comp_info['Attribute'].valuesdef remove_duplicates(x):
    return list(dict.fromkeys(x))def clear_df(the_list):
    the_list = (the_list.split('\n')[1].split('  '))
    the_list = [value for value in the_list if value != ''][0]
    return the_listdef ultra_clear_df(the_list):
    the_list = (the_list.split('\n\n')[1].split('  '))
    the_list = [value for value in the_list if value != ''][0]
    the_list = (the_list.split('\n')[0])
    return the_list
```

啊，我们走吧！
我们刚刚定义了 5 个函数:

*   connect():用于连接到网站，并从网站下载原始 html 代码；
*   get_areas():它抓取原始 html 来查找地区。每个地区都有一个唯一的链接，过滤与该地区相关的公告；
*   get_pages():对于每个地区的“主页”，它查找有多少页的公告可用，并为每一页创建一个链接；
*   get_apartment_links():对于找到的每个页面，它查找每个公告并收集每个链接
*   scrape_link():这个函数是 announces 的正确抓取过程

在执行的最后，我们将会有每一个公告的链接，并注明来源地区。

```
*## Get areas inside the city (districts)*

website = "https://www.immobiliare.it/affitto-case/torino"
districts = get_areas(website)
print("Those are district's links **\n**")
print(districts)## Now we need to find all announces' links, in order to scrape informations inside them one by oneaddress = []
location = []try:
    for url in tqdm(districts):
        pages = get_pages(url)
        for page in pages:
            add = get_apartment_links(page)
            address.append(add)
            for num in range(0,len(add)):
                location.append(url.rsplit('/', 1)[-1])
except Exception as e:
    print(e)
    continue

announces_links = [item for value in address for item in value]## Just check it has some sense and save itprint("The numerosity of announces:**\n**")
print(len(announces_links))
**with** open('announces_list.csv', 'w') **as** myfile:
    wr = csv.writer(myfile)
    wr.writerow(announces_links)
```

现在我们有了那个特定城市的每个公告的链接，所以让我们寻找宝藏。

# 工作中的厨师！

```
## Now we pass all announces' links do the scrape_link function to obtain apartments' informationsdf_scrape = pd.DataFrame()
to_be_dropped = []
counter = 0
for link in tqdm(list(announces_links)):
    counter=counter+1
    try:
        names, values = scrape_link(link)
        temp_df = pd.DataFrame(columns=names)
        temp_df.loc[len(temp_df), :] = values[0:len(names)]
        df_scrape = df_scrape.append(temp_df, sort=False)
    except Exception as e:
        print(e)
        to_be_dropped.append(counter)
        print(to_be_dropped)
        continue## Eventually save useful informations odtained during the scrape processpd.DataFrame(location).to_csv('location.csv', sep=';')
pd.DataFrame(to_be_dropped).to_csv('to_be_dropped.csv', sep=';')
```

这段代码遍历每个公告并从中提取信息，收集两个列表中的所有信息: *nomi* 和 *valori。*第一个包含特征的名称，第二个包含值。
在最后的刮削过程中，我们终于有了一只熊猫。一种数据帧，其中每一行都存储着一个不同的通告及其特征和所属的地区。
只要检查出精细的数据帧就有意义。

```
print(df_scrape.shape)
df_scrape[‘district’] = location
df_scrape[‘links’] = announces_links
df_scrape.columns = map(str.lower, df_scrape.columns)
df_scrape.to_csv(‘dataset.csv’, sep=”;”)
```

现在我们有了一个包含 24 列(24 个特征)的数据框架，我们可以用它来训练我们的回归算法。

现在，在把所有的东西放进锅里之前，我们必须清理干净并把配料切片…

# 切片、切割和清理

不幸的是，数据集并不完全…嗯…准备好了。我们收集的东西通常很脏，我们无法处理。仅举几个例子:价格以“600€/月”的形式存储为*字符串*，5 间以上的房子列为*“6+”*，等等。

所以在这里我们有工具来“清洗他们所有人”(“咕鲁，咕鲁！”)

```
df_scrape = df_scrape[['contratto', 'zona', 'tipologia', 'superficie', 'locali', 'piano', 'tipo proprietà', 'prezzo', 'spese condominio', 'spese aggiuntive', 'anno di costruzione', 'stato', 'riscaldamento', 'climatizzazione', 'posti auto', 'links']]def cleanup(df):
    price = []
    rooms = []
    surface = []
    bathrooms = []
    floor = []
    contract = []
    tipo = []
    condominio = []
    heating = []
    built_in = []
    state = []
    riscaldamento = []
    cooling = []
    energy_class = []
    tipologia = []
    pr_type = []
    arredato = []

    for tipo in df['tipologia']:
        try:
            tipologia.append(tipo)
        except:
            tipologia.append(None)

    for superficie in df['superficie']:
        try:
            if "m" in superficie:
                #z = superficie.split('|')[0]
                s = superficie.replace(" m²", "")
                surface.append(s)
        except:
            surface.append(None)

    for locali in df['locali']:
        try:
            rooms.append(locali[0:1])
        except:
            rooms.append(None)

    for prezzo in df['prezzo']:
        try:
            price.append(prezzo.replace("Affitto ", "").replace("€ ", "").replace("/mese", "").replace(".",""))
        except:
            price.append(None)

    for contratto in df['contratto']:
        try:
            contract.append(contratto.replace("\n ",""))
        except:
            contract.append(None)

    for piano in df['piano']:
        try:
            floor.append(piano.split(' ')[0])
        except:
            floor.append(None)

    for tipologia in df['tipo proprietà']:
        try:
            pr_type.append(tipologia.split(',')[0])
        except:
            pr_type.append(None)

    for condo in df['spese condominio']:
        try:
            if "mese" in condo:
                condominio.append(condo.replace("€ ","").replace("/mese",""))
            else:
                condominio.append(None)
        except:
            condominio.append(None)

    for ii in df['spese aggiuntive']:
        try:
            if "anno" in ii:
                mese = int(int(ii.replace("€ ","").replace("/anno","").replace(".",""))/12)
                heating.append(mese)
            else:
                heating.append(None)
        except:
            heating.append(None)

    for anno_costruzione in df['anno di costruzione']:
        try:
            built_in.append(anno_costruzione)
        except:
            built_in.append(None)

    for stato in df['stato']:
        try:
            stat = stato.replace(" ","").lower()
            state.append(stat)
        except:
            state.append(None)

    for tipo_riscaldamento in df['riscaldamento']:
        try:
            if 'Centralizzato' in tipo_riscaldamento:
                riscaldamento.append('centralizzato')
            elif 'Autonomo' in tipo_riscaldamento:
                riscaldamento.append('autonomo')
        except:
            riscaldamento.append(None)

    for clima in df['climatizzazione']:
        try:
            cooling.append(clima.lower().split(',')[0])
        except:
            cooling.append('None')

    final_df = pd.DataFrame(columns=['contract', 'district', 'renting_type', 'surface', 'locals', 'floor', 'property_type', 'price', 'spese condominio', 'other_expences', 'building_year', 'status', 'heating', 'air_conditioning', 'energy_certificate', 'parking_slots'])#, 'Arredato S/N'])
    final_df['contract'] = contract
    final_df['renting_type'] = tipologia
    final_df['surface'] = surface
    final_df['locals'] = rooms
    final_df['floor'] = floor
    final_df['property_type'] = pr_type
    final_df['price'] = price
    final_df['spese condominio'] = condominio
    final_df['heating_expences'] = heating
    final_df['building_year'] = built_in
    final_df['status'] = state
    final_df['heating_system'] = riscaldamento
    final_df['air_conditioning'] = cooling
    #final_df['classe energetica'] = energy_class
    final_df['district'] = df['zona'].values
    #inal_df['Arredato S/N'] = arredato
    final_df['announce_link'] = announces_links

    return final_dffinal = cleanup(df_scrape)
final.to_csv('regression_dataset.csv', sep=";")
```

该函数根据脏数据的类型，以不同的方式处理脏数据。他们中的大多数已经用 Regex**和 string 相关的工具清理过了。看看这个脚本，看看它是如何工作的。
*PS:数据集上可能仍然存在少量错误。像你喜欢的那样对待他们。***

# 准备锅！

我们到了！现在我们有了一个手工制作的数据集，我们可以在其上处理 ML 和回归。在下一篇文章中，我将解释我如何处理所有的成分。

敬请期待！

*链接 GitHub*[*https://github.com/wonka929/house_scraping_and_regression*](https://github.com/wonka929/house_scraping_and_regression)

这篇文章是教程的第一部分。你可以在这个链接找到第二篇:
[https://medium . com/@ wonka 929/house-rent-the-data-science-way-part-2-train-a-regression-model-tpot-and-auto-ml-9 CDB 5 CB 4 B1 b 4](/@wonka929/house-rental-the-data-science-way-part-2-train-a-regression-model-tpot-and-auto-ml-9cdb5cb4b1b4)

***更新*** *:在处理 immobiliare.it 的新网站时，我决定也更新一下回归方法论。
这是您可以在网上找到的新的更新文章:* [https://wonka 929 . medium . com/house-rent-the-data-science-way-part-2-1-train-and-regression-model-using-py caret-72d 054 e22a 78](https://wonka929.medium.com/house-rental-the-data-science-way-part-2-1-train-and-regression-model-using-pycaret-72d054e22a78)