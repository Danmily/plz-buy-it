"""挽单评测与训练数据的场景库。

test / train 两个 split 使用不重叠的商品，且每组表述按奇偶位拆分（test 取偶数位，train 取奇数位），
所以用 train split 生成的训练数据不会把测试集里的商品和说法泄漏给模型。
"""
import copy
import random

CONCERNS = ['价格', '等大促', '凑单', '比价', '品质信任', '规格适配', '物流售后', '流程摩擦']
ACTIONS = ['定向券', '价格事实', '分期信息', '价保', '凑单推荐', '对比卡片', '好评与保障',
           '尺码与规格助手', '时效与运费险', '支付引导', '无']
TOUCHPOINTS = ['页内', '弹窗', 'push', '再曝光', '对话回复', '无']
ACTION_OK = {
    '价格': ['定向券', '价格事实'], '等大促': ['价保'], '凑单': ['凑单推荐'], '比价': ['对比卡片'],
    '品质信任': ['好评与保障'], '规格适配': ['尺码与规格助手'], '物流售后': ['时效与运费险'], '流程摩擦': ['支付引导'],
}
TOUCH_OK = {
    '价格': ['页内', 'push'], '等大促': ['push', '页内'], '凑单': ['页内'], '比价': ['再曝光', '页内'],
    '品质信任': ['页内', '再曝光'], '规格适配': ['页内'], '物流售后': ['页内'], '流程摩擦': ['页内'],
}
NO_ACT_TEXT = {
    '必买': '复购周期内直达常买款，没有犹豫行为，不需要挽留',
    '弱需求': '停留很短、没有深度行为，挽留只会打扰',
    '已购替代': '本次会话已买了同类替代品，应停止挽留',
    '频控': '近期多次挽留都被关闭，进入冷却期',
    '敏感类目': '敏感类目不做挽留',
    '囤货型': '购物车长期囤货、很少结算，不主动挽留',
    '用户拒绝': '用户在对话中明确拒绝，尊重用户意愿',
    '非购物意图': '用户在咨询已有订单，不是购买犹豫',
}
SHOPPING_CATS = ['服饰鞋包', '3C数码', '美妆个护', '家电']
SENSITIVE_CATS = ['药品', '母婴']


def _it(name, price, avg, low, rating, ship, kind, opts):
    return dict(name=name, price=price, avg=avg, low=low, rating=rating, ship=ship, kind=kind, opts=opts)


ITEMS = {
    'test': {
        '服饰鞋包': [_it('轻量缓震跑鞋', 529, 539, 499, 4.7, '48 小时发货', '鞋码', ['40', '41', '42']),
                 _it('羊毛混纺大衣', 899, 869, 799, 4.5, '72 小时发货', '尺码', ['S', 'M', 'L']),
                 _it('通勤双肩包', 259, 249, 229, 4.8, '24 小时发货', '颜色', ['黑色', '灰色']),
                 _it('加绒卫衣', 199, 189, 169, 4.6, '48 小时发货', '尺码', ['M', 'L', 'XL'])],
        '3C数码': [_it('降噪蓝牙耳机', 399, 372, 359, 4.6, '24 小时发货', '版本', ['标准版', 'Pro 版']),
                 _it('65W 氮化镓充电器', 129, 125, 109, 4.8, '24 小时发货', '颜色', ['白色', '黑色']),
                 _it('轻薄笔记本电脑', 5999, 5899, 5699, 4.7, '48 小时发货', '配置', ['16G+512G', '32G+1T']),
                 _it('智能手表', 1299, 1259, 1199, 4.5, '48 小时发货', '表盘尺寸', ['41mm', '45mm'])],
        '美妆个护': [_it('丝绒口红', 189, 179, 159, 4.7, '24 小时发货', '色号', ['#01 正红', '#05 豆沙']),
                 _it('保湿精华液', 459, 439, 399, 4.6, '48 小时发货', '容量', ['30ml', '50ml'])],
        '家电': [_it('65 英寸智能电视', 4299, 4199, 3999, 4.6, '72 小时发货，需预约安装', '尺寸', ['55 英寸', '65 英寸']),
               _it('洗烘一体机', 3699, 3599, 3399, 4.5, '72 小时发货，需预约安装', '容量', ['10kg', '12kg'])],
        '快消': [_it('浓缩洗衣液 3kg', 59.9, 59.9, 55.9, 4.9, '次日达', '规格', ['3kg']),
               _it('抽纸 24 包', 49.9, 52.9, 45.9, 4.8, '次日达', '规格', ['24 包'])],
        '药品': [_it('感冒颗粒 10 袋', 29.9, 29.9, 27.9, 4.7, '24 小时发货', '规格', ['10 袋'])],
        '母婴': [_it('婴儿配方奶粉 3 段', 268, 259, 239, 4.8, '48 小时发货', '规格', ['800g'])],
    },
    'train': {
        '服饰鞋包': [_it('复古帆布鞋', 299, 309, 279, 4.6, '48 小时发货', '鞋码', ['38', '39', '40']),
                 _it('法兰绒衬衫', 239, 229, 199, 4.5, '48 小时发货', '尺码', ['M', 'L', 'XL']),
                 _it('真皮托特包', 1199, 1159, 1099, 4.7, '72 小时发货', '颜色', ['棕色', '黑色']),
                 _it('摇粒绒外套', 329, 319, 289, 4.6, '48 小时发货', '尺码', ['S', 'M', 'L'])],
        '3C数码': [_it('头戴式降噪耳机', 899, 869, 829, 4.7, '24 小时发货', '颜色', ['银色', '黑色']),
                 _it('磁吸无线充电宝', 199, 189, 169, 4.6, '24 小时发货', '容量', ['5000mAh', '10000mAh']),
                 _it('游戏本', 6999, 6899, 6599, 4.6, '48 小时发货', '配置', ['RTX 4060', 'RTX 4070']),
                 _it('运动手环', 249, 239, 219, 4.5, '24 小时发货', '版本', ['标准版', 'NFC 版'])],
        '美妆个护': [_it('持妆粉底液', 329, 319, 289, 4.6, '24 小时发货', '色号', ['自然色', '象牙白']),
                 _it('修护面霜', 589, 569, 529, 4.7, '48 小时发货', '容量', ['50ml', '100ml'])],
        '家电': [_it('扫地机器人', 2499, 2399, 2299, 4.6, '48 小时发货', '版本', ['标准版', '自动集尘版']),
               _it('空气净化器', 1599, 1549, 1499, 4.5, '48 小时发货', '适用面积', ['40㎡', '60㎡'])],
        '快消': [_it('洗洁精 1.5kg', 19.9, 21.9, 17.9, 4.8, '次日达', '规格', ['1.5kg']),
               _it('牙膏 4 支装', 39.9, 42.9, 35.9, 4.8, '次日达', '规格', ['4 支'])],
        '药品': [_it('维生素 C 泡腾片', 59, 59, 52, 4.7, '24 小时发货', '规格', ['20 片'])],
        '母婴': [_it('纸尿裤 L 码', 129, 125, 115, 4.8, '48 小时发货', '规格', ['L 码 52 片'])],
    },
}
ALT_BRANDS = {'test': ['悦声', '极光', '知行'], 'train': ['轻域', '木棉', '启明']}
NOISE_ITEMS = ['保温杯', '手机壳', '零食大礼包', '桌面收纳盒', '运动袜 5 双', '数据线']
CITIES = ['拉萨', '乌鲁木齐', '喀什', '西宁']

# ---- 表述库：每组至少 2 个说法，偶数位给 test，奇数位给 train ----
SIG = {
    '价格': [
        ('商详', ['展开优惠明细，领取 ¥{coupon} 店铺券但未使用', '点开「优惠」浮层，领了 ¥{coupon} 券没有下单', '查看到手价计算明细，停留 {s} 秒']),
        ('商详', ['离开后再次回到该商品，停留在价格区 {s} 秒', '第 {k} 次打开该商品，主要停在价格区域', '再次进入商详，放大查看价格标签']),
        ('搜索', ['搜索「{name} 降价」', '搜索「{name} 优惠券」', '搜索「{name} 便宜」']),
        ('商详', ['点击「降价提醒」按钮', '订阅该商品的降价通知', '点击「到手价提醒」']),
        ('购物车', ['把该商品数量从 2 件改成 1 件', '勾选该商品看合计金额后又取消勾选', '在购物车里查看该商品的「比加入时」价格变化']),
    ],
    '等大促': [
        ('搜索', ['搜索「双11 {name}」', '搜索「{name} 活动价」', '搜索「{name} 什么时候打折」']),
        ('会场', ['浏览双11 大促会场预告页 {s} 秒', '进入「预售」频道浏览 {s} 秒', '查看平台大促活动日历']),
        ('商详', ['查看「大促到手价」预告浮层', '点开商详上的「11.11 预告价」标签', '查看「定金膨胀」规则']),
        ('商详', ['点击「开售提醒」', '预约该商品的大促开抢提醒', '把该商品加入「大促清单」']),
    ],
    '凑单': [
        ('购物车', ['查看店铺满减规则（满 300 减 30）', '点开「满减」标签查看门槛', '查看「跨店满减」说明']),
        ('购物车', ['反复勾选、取消购物车里的另一件小商品', '在购物车里来回调整其他商品数量，观察合计金额', '勾选又取消一件 ¥{gap} 左右的小商品']),
        ('凑单页', ['进入「凑单专区」浏览 {s} 秒后返回购物车', '点击「去凑单」，浏览 {n} 个商品后返回', '在凑单页按价格从低到高排序']),
    ],
    '比价': [
        ('商详', ['进入同类商品「{alt}」（¥{altp}）商详，停留 {s} 秒', '点开推荐位里的「{alt}」（¥{altp}），看了 {s} 秒', '从「看了又看」进入「{alt}」商详']),
        ('搜索', ['搜索「{name} 和 {alt} 哪个好」', '搜索「{name} 平替」', '搜索「{name} 对比」']),
        ('商详', ['在两款同类商品商详之间来回切换 {k} 次', '返回搜索结果页，依次点开 {k} 个同类商品', '把两款同类商品都加入「对比」']),
        ('购物车', ['把「{alt}」也加入购物车', '购物车里同时放着「{alt}」和该商品', '将「{alt}」加入购物车后又回到该商品']),
    ],
    '品质信任': [
        ('商详', ['评价区筛选「差评」，浏览 {n} 条', '评价区按「最新」排序，翻看低分评价 {n} 条', '点开评价区「有图差评」']),
        ('商详', ['查看买家秀图片 {n} 张', '放大查看买家实拍视频 {s} 秒', '翻看「追加评价」{n} 条']),
        ('店铺', ['点击店铺资质与店铺评分', '查看店铺「品牌授权」证书', '进入店铺首页查看开店时长']),
        ('商详', ['在「问大家」搜索「是不是正品」', '在「问大家」查看「质量怎么样」的回答', '在「问大家」提问「用久了会不会坏」']),
    ],
    '规格适配': [
        ('商详', ['选择{kind} {a} → 切换为 {b} → 又切回 {a}', '在{kind} {a} 和 {b} 之间来回切换 {k} 次', '先选{kind} {b}，几秒后改成 {a}，又改回 {b}']),
        ('商详', ['打开{kind}对照表，停留 {s} 秒', '使用「{kind}推荐」工具并输入个人数据', '点开「{kind}说明」图片放大查看']),
        ('商详', ['评价区筛选「{kind}」相关评价', '在评价区搜索「偏大」「偏小」', '翻看带有身高体重或使用场景的评价 {n} 条']),
        ('商详', ['在「问大家」提问「{kind}怎么选」', '在「问大家」查看「{kind}准不准」', '在「问大家」搜索「{a} 还是 {b}」']),
    ],
    '物流售后': [
        ('商详', ['查看配送时效，显示预计 {days} 天送达', '点开「配送至」浮层查看送达时间', '查看「发货时间」说明']),
        ('商详', ['查看退换货与运费险说明', '查看「7 天无理由退货」规则', '查看「保修与售后」说明']),
        ('结算', ['把收货地址切换到{city}', '在结算页新增{city}的收货地址', '结算页收货地址为{city}，点开运费明细']),
        ('商详', ['在「问大家」搜索「多久能到」', '在「问大家」查看「发什么快递」', '在「问大家」查看「退货麻不麻烦」']),
    ],
    '流程摩擦': [
        ('结算', ['提交订单后支付失败（银行卡验证未通过）', '支付时提示「银行卡余额不足」', '支付页加载超时，未完成支付']),
        ('结算', ['切换支付方式后再次支付失败', '更换为另一张银行卡后仍提示失败', '尝试分期付款，提示额度不足']),
        ('结算', ['修改收货地址，提示超出配送范围', '优惠券无法叠加，提示「不满足使用条件」', '结算页提示「部分商品库存不足」']),
        ('结算', ['结算页停留 {s} 秒，反复点击支付按钮', '在支付页反复返回、重进 {k} 次', '结算页刷新 {k} 次']),
    ],
}
POST_PAGES = {'购物车', '结算', '凑单页'}

# 看起来像信号、其实不说明任何顾虑的行为
NEUTRAL = [
    ('商详', ['查看商品参数', '浏览商品详情图 {n} 张', '查看商品规格参数表']),
    ('商详', ['查看评价区，浏览 {n} 条好评', '点开评价区的「好评」标签', '看了几条带图好评']),
    ('商详', ['收藏该商品', '把商品分享给好友', '截图商品页']),
    ('系统', ['App 切到后台 {m} 分钟后返回', '锁屏 {m} 分钟后回到 App', '接听电话，{m} 分钟后返回']),
    ('推荐流', ['再次刷到该商品的推荐卡片', '在首页再次看到该商品，未点击', '刷到同店其他商品的短视频']),
]
NOISE = [
    ('首页', ['浏览首页推荐 {s} 秒', '刷首页信息流 {s} 秒']),
    ('商详', ['点进无关商品「{other}」，停留 {m} 秒', '顺手看了一眼「{other}」']),
    ('订单', ['查看一笔已有订单的物流', '查看历史订单列表']),
    ('短视频', ['观看短视频 {s} 秒', '连续刷了 {k} 条短视频']),
]
ENTRY = {
    '搜索': ('搜索', ['搜索「{name}」后进入商详', '在搜索结果第 {k} 位点进「{name}」', '搜索「{name}」，从结果页进入商详']),
    '推荐流': ('推荐流', ['从首页推荐点进「{name}」商详', '在「猜你喜欢」点进「{name}」', '从推荐信息流点进「{name}」商详']),
    '短视频': ('短视频', ['从短视频商品卡进入「{name}」商详', '看完一条测评短视频后点击商品链接进入「{name}」', '在短视频评论区点击「{name}」链接']),
    '直播间': ('直播间', ['在直播间点击商品链接进入「{name}」商详', '直播讲解时点击小黄车进入「{name}」', '从直播间购物袋进入「{name}」商详']),
    'push 召回': ('消息', ['点击 push 消息进入「{name}」商详', '从站内信进入「{name}」商详', '点击「你关注的商品有更新」消息进入「{name}」']),
    '店铺': ('店铺', ['在店铺首页点进「{name}」', '从店铺分类页进入「{name}」商详', '在关注店铺的上新里点进「{name}」']),
}
BROWSE = ['浏览商详 {s} 秒', '上下滑动浏览商品详情 {s} 秒', '在商详停留 {s} 秒']
EXIT = {
    '商详': ('商详', ['离开商详，未加购', '返回上一页，没有加购'], '商详离开未加购'),
    '商详加购': ('商详', ['加购后离开商详', '加入购物车后返回首页'], '加购后离开'),
    '购物车': ('购物车', ['离开购物车，未勾选结算', '退出购物车页面，没有结算'], '购物车离开'),
    '结算': ('结算', ['退出结算页，订单未支付', '关闭支付页，订单待支付'], '结算页退出'),
}

CHAT = {
    '价格': {'explicit': ['能便宜点吗', '还有没有优惠券', '这个价格有点贵了', '现在买有什么优惠'],
           'implicit': ['学生党预算有限', '上次看好像没这么贵', '有没有更划算的买法', '等等看会不会便宜']},
    '等大促': {'explicit': ['双十一会更便宜吗', '要不要等活动再买', '大促有没有保价', '11.11 是什么价'],
            'implicit': ['最近好像快有活动了吧', '过几天会不会有活动', '听说下周有大促']},
    '凑单': {'explicit': ['满减怎么算', '还差多少能用满减', '跨店满减能叠加吗', '满 300 减 30 怎么凑'],
           'implicit': ['再买点什么划算', '凑一凑能省多少', '加点什么能更便宜']},
    '比价': {'explicit': ['这款和{alt}哪个好', '和{alt}比有什么区别', '帮我对比一下这款和{alt}', '{alt}便宜一些，差在哪'],
           'implicit': ['别家好像也有类似的', '我还在看别的款', '还有没有同类的推荐']},
    '品质信任': {'explicit': ['是正品吗', '质量怎么样，看到有差评', '会不会用几天就坏', '有没有质检报告'],
             'implicit': ['之前买过类似的用几天就坏了', '这个牌子没怎么听过', '店铺好像是新开的']},
    '规格适配': {'explicit': ['{kind}选 {a} 还是 {b}', '{kind}{a} 合适吗', '{kind}怎么选比较好', '{a} 和 {b} 差别大吗'],
             'implicit': ['之前买的{kind}不合适退过一次', '给家里人买，不确定{kind}', '平时用 {a} 有点不够']},
    '物流售后': {'explicit': ['几天能到{city}', '能发顺丰吗', '退货运费谁出', '坏了怎么保修'],
             'implicit': ['下周要出差，怕收不到', '之前退货很麻烦', '家里白天没人收快递']},
    '流程摩擦': {'explicit': ['怎么付不了款', '地址填不上', '券用不了是怎么回事', '一直提示支付失败'],
             'implicit': ['点了支付没反应', '页面一直转圈', '弄了半天没付成功']},
}
CHAT_OTHER = {
    'think': ['我再想想', '先看看', '嗯，再考虑一下', '晚点再说'],
    'optout': ['不需要了，别推了', '先不买了，不用再提醒我', '别发消息了', '不用了，谢谢，别再推送'],
    'nonshop': ['我上一单到哪了', '怎么开发票', '上一单能改地址吗', '我的退款什么时候到'],
}
BOT = ['您好，请问有什么可以帮您？', '在的，请问有什么问题？', '您好，我是智能客服']
BOT_REPLY = ['好的，已为您记录', '稍等，正在为您查询', '收到，还有其他问题吗？', '已收到您的问题']

RANDS = {'s': (8, 90), 'n': (4, 25), 'k': (2, 5), 'm': (1, 15), 'days': (4, 7)}


class _Fmt(dict):
    def __init__(self, r, base):
        super().__init__(base)
        self.r = r

    def __missing__(self, key):
        if key in RANDS:
            return self.r.randint(*RANDS[key])
        if key == 'coupon':
            return self.r.choice([5, 10, 20])
        if key == 'city':
            return self.r.choice(CITIES)
        if key == 'other':
            return self.r.choice(NOISE_ITEMS)
        raise KeyError(key)


def pick(r, variants, split):
    return r.choice(variants[0::2] if split == 'test' else variants[1::2])


def fmt_price(p):
    return f'{p:g}'


def pick_item(r, split, cats=None, need_opts=False, min_price=0, max_price=1e9):
    cats = cats or SHOPPING_CATS
    pool = [(c, it) for c in cats for it in ITEMS[split][c]
            if (not need_opts or len(it['opts']) > 1) and min_price <= it['price'] <= max_price]
    cat, it = r.choice(pool)
    return cat, copy.deepcopy(it)


def item_for(r, split, c):
    if c == '规格适配':
        return pick_item(r, split, need_opts=True)
    if c == '凑单':
        return pick_item(r, split, max_price=290)
    return pick_item(r, split)


def item_base(r, split, it):
    o = it['opts']
    a, b = (o[0], o[1]) if len(o) > 1 else (o[0], o[0])
    return dict(name=it['name'], kind=it['kind'], a=a, b=b, alt=r.choice(ALT_BRANDS[split]) + ' ' + it['name'],
                altp=round(it['price'] * 0.85), gap=r.randint(10, 40))


def make_profile(r, tier=None, price_sens=None, extras=()):
    return dict(tier=tier or r.choice(['高活跃', '中活跃', '低活跃', '新客']),
                price_sens=price_sens or r.choice(['高', '中', '低']), extras=list(extras))


class Session:
    def __init__(self, r, split, cat, it):
        self.r, self.split, self.cat, self.it = r, split, cat, it
        self.base = item_base(r, split, it)
        self.ev, self.extra, self.multi = [], [], False

    def fmt(self, tpl, base=None):
        return tpl.format_map(_Fmt(self.r, base or self.base))

    def e(self, page, desc, c=None, item='focus'):
        return dict(page=page, desc=desc, c=c, item=self.it['name'] if item == 'focus' else item)

    def signals(self, c, n, base=None, item='focus'):
        tpls = self.r.sample(SIG[c], min(n, len(SIG[c])))
        return [self.e(p, self.fmt(pick(self.r, vs, self.split), base), c, item) for p, vs in tpls]

    def lib(self, lib):
        p, vs = self.r.choice(lib)
        return self.e(p, self.fmt(pick(self.r, vs, self.split)), None, None)

    def chat(self, utts, page='商详'):
        r, sp = self.r, self.split
        out = [self.e(page, '点击「客服」进入对话'), self.e('对话', f'客服机器人：「{pick(r, BOT, sp)}」', None, None)]
        for text, c in utts:
            out.append(self.e('对话', f'用户：「{self.fmt(text)}」', c))
            out.append(self.e('对话', f'客服机器人：「{pick(r, BOT_REPLY, sp)}」', None, None))
        return out


def flow(s, groups, stage, *, noise_k=0, neutral_k=0, entry=None, others=(), chat=None, chat_pos='pre',
         first_events=()):
    """拼一段会话：入口 → 商详 →（其他商品）→ 前置信号 →（对话）→ 加购/购物车/结算 → 后置信号 → 离开。"""
    r = s.r
    entry = entry or r.choice(list(ENTRY))
    page, vs = ENTRY[entry]
    s.ev.append(s.e(page, s.fmt(pick(r, vs, s.split))))
    s.ev += list(first_events)
    s.ev.append(s.e('商详', s.fmt(pick(r, BROWSE, s.split))))
    for oit, oc, on in others:
        s.multi = True
        ob = item_base(r, s.split, oit)
        s.ev.append(s.e('商详', f'进入「{oit["name"]}」商详', None, oit['name']))
        s.ev.append(s.e('商详', s.fmt(pick(r, BROWSE, s.split)), None, oit['name']))
        if oc:
            s.ev += s.signals(oc, on, base=ob, item=oit['name'])
        s.ev.append(s.e('商详', f'回到「{s.it["name"]}」商详'))
    sig = []
    for c, n in groups:
        sig += s.signals(c, n)
    r.shuffle(sig)
    pre = [x for x in sig if x['page'] not in POST_PAGES]
    post = [x for x in sig if x['page'] in POST_PAGES]
    for _ in range(neutral_k):
        pre.insert(r.randint(0, len(pre)), s.lib(NEUTRAL))
    for _ in range(noise_k):
        pre.insert(r.randint(0, len(pre)), s.lib(NOISE))
    s.ev += pre
    if chat and chat_pos == 'pre':
        s.ev += s.chat(chat)
    final = stage
    if stage == '商详加购':
        s.ev.append(s.e('商详', '加入购物车'))
    if stage in ('购物车', '结算') or post:
        if stage != '商详加购':
            s.ev.append(s.e('商详', '加入购物车'))
        s.ev.append(s.e('购物车', s.fmt('进入购物车，停留 {s} 秒')))
        if stage == '结算' or any(x['page'] == '结算' for x in post):
            s.ev.append(s.e('结算', '进入结算页'))
            final = '结算'
        elif stage in ('商详', '商详加购'):
            final = '购物车'
    s.ev += post
    if chat and chat_pos == 'post':
        s.ev += s.chat(chat, page='结算' if final == '结算' else '购物车')
    p, vs, dp = EXIT[final]
    s.ev.append(s.e(p, pick(r, vs, s.split)))
    return dp


def stage_for(c, r):
    if c == '流程摩擦':
        return '结算'
    if c == '凑单':
        return '购物车'
    if c == '物流售后':
        return r.choice(['商详', '结算'])
    return r.choice(['商详', '商详加购', '购物车'])


def extra_for(c, r, it):
    if c == '等大促':
        return [f'距双11 开售还有 {r.randint(3, 9)} 天']
    if c == '凑单':
        return ['店铺满 300 减 30', f'购物车合计 ¥{r.randint(262, 292)}']
    if c == '物流售后':
        return ['支持运费险']
    if c == '价格' and r.random() < 0.5:
        return [f'该用户 3 天内第 {r.randint(3, 6)} 次访问该商品']
    if c == '品质信任' and r.random() < 0.4:
        return [f'店铺开店 {r.randint(2, 8)} 个月，店铺评分 4.3']
    return []


def g_act(primary, secondary_ok=(), action_ok=None, must_not=(), touch_ok=None, allow_uncertain=False, channel='app'):
    must = list(must_not)
    if primary != '价格' and '定向券' not in must:
        must.append('定向券')
    return dict(act=True, act_reason='顾虑可挽', primary=primary, secondary_ok=list(secondary_ok),
                action_ok=list(action_ok or ACTION_OK[primary]), must_not=must,
                touch_ok=list(touch_ok or (['对话回复'] if channel == 'chat' else TOUCH_OK[primary])),
                allow_uncertain=allow_uncertain, score_primary=True)


def g_no(reason, primary='无', secondary_ok=(), score_primary=False):
    return dict(act=False, act_reason=reason, primary=primary, secondary_ok=list(secondary_ok), action_ok=['无'],
                must_not=['定向券'], touch_ok=['无'], allow_uncertain=False, score_primary=score_primary)


def finalize(s, cid, gold, profile, dp, layer, scenario, tags=(), channel='app'):
    r = s.r
    h, m = r.randint(8, 23), r.randint(0, 30)
    events, ev_by = [], {}
    for i, x in enumerate(s.ev, 1):
        m += r.randint(0, 2)
        events.append(dict(i=i, ts=f'{h:02d}:{min(m, 59):02d}', page=x['page'], desc=x['desc'], item=x['item']))
        if x['c'] and x['item'] == s.it['name']:
            ev_by.setdefault(x['c'], []).append(i)
    gold = dict(gold)
    gold['evidence'] = ev_by.get(gold['primary'], []) if gold['primary'] != '无' else []
    gold['evidence_by_concern'] = ev_by
    it = s.it
    item = dict(category=s.cat, name=it['name'], price=it['price'], avg=it['avg'], low=it['low'],
                rating=it['rating'], ship=it['ship'], extra=s.extra)
    return dict(id=cid, scenario=scenario, layer=layer, channel=channel, tags=list(tags), multi_item=s.multi,
                profile=profile, item=item, events=events, decision_point=dp, gold=gold)


# ---------------- 场景 ----------------

def sc_app_clear(r, sp, cid, c):
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(c, r, it)
    prof = make_profile(r, price_sens='高' if c == '价格' else None)
    dp = flow(s, [(c, r.randint(2, 3))], stage_for(c, r), neutral_k=r.randint(0, 1))
    return finalize(s, cid, g_act(c), prof, dp, 'L1_清晰', f'app_clear:{c}')


def sc_app_noisy(r, sp, cid, c):
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(c, r, it)
    groups, sec = [(c, r.randint(2, 3))], []
    if r.random() < 0.5:
        o = r.choice([x for x in CONCERNS if x not in (c, '流程摩擦', '凑单', '规格适配')])
        groups.append((o, 1))
        sec = [o]
    others = []
    if r.random() < 0.3:
        o_it = pick_item(r, sp, cats=[cat])[1]
        if o_it['name'] != it['name']:
            others = [(o_it, None, 0)]
    dp = flow(s, groups, stage_for(c, r), noise_k=r.randint(3, 6), neutral_k=r.randint(2, 3), others=others)
    return finalize(s, cid, g_act(c, secondary_ok=sec), make_profile(r), dp, 'L2_干扰', f'app_noisy:{c}')


def sc_app_sparse(r, sp, cid, c):
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(c, r, it)
    dp = flow(s, [(c, 1)], stage_for(c, r))
    return finalize(s, cid, g_act(c, allow_uncertain=True), make_profile(r, tier=r.choice(['低活跃', '新客'])),
                    dp, 'L4_稀疏', f'app_sparse:{c}')


DUAL_OK = [x for x in CONCERNS if x not in ('流程摩擦', '凑单')]


def sc_app_dual(r, sp, cid):
    a, b = r.sample(DUAL_OK, 2)
    cat, it = item_for(r, sp, '规格适配' if '规格适配' in (a, b) else a)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(a, r, it) + extra_for(b, r, it)
    dp = flow(s, [(a, 3), (b, 1)], stage_for(a, r), noise_k=r.randint(0, 2), neutral_k=r.randint(0, 2))
    return finalize(s, cid, g_act(a, secondary_ok=[b]), make_profile(r), dp, 'L3_双顾虑', 'app_dual', [f'次:{b}'])


def sc_chat_explicit(r, sp, cid, c):
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(c, r, it)
    pos = 'post' if c in ('流程摩擦', '凑单') else 'pre'
    utt = [(pick(r, CHAT[c]['explicit'], sp), c)]
    dp = flow(s, [(c, r.randint(0, 1))], stage_for(c, r), chat=utt, chat_pos=pos, noise_k=r.randint(0, 2))
    return finalize(s, cid, g_act(c, channel='chat'), make_profile(r), dp, 'C_对话', f'chat_explicit:{c}',
                    channel='chat')


def sc_chat_implicit(r, sp, cid, c):
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(c, r, it)
    pos = 'post' if c in ('流程摩擦', '凑单') else 'pre'
    utt = [(pick(r, CHAT[c]['implicit'], sp), c)]
    dp = flow(s, [(c, 1)], stage_for(c, r), chat=utt, chat_pos=pos, noise_k=r.randint(0, 2))
    return finalize(s, cid, g_act(c, channel='chat'), make_profile(r), dp, 'C_对话', f'chat_implicit:{c}',
                    channel='chat')


def sc_chat_conflict(r, sp, cid):
    x, y = r.sample(DUAL_OK, 2)  # 行为显示 x，用户亲口说 y
    cat, it = item_for(r, sp, '规格适配' if '规格适配' in (x, y) else y)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(y, r, it)
    dp = flow(s, [(x, 2)], stage_for(y, r), chat=[(pick(r, CHAT[y]['explicit'], sp), y)])
    return finalize(s, cid, g_act(y, secondary_ok=[x], channel='chat'), make_profile(r), dp, 'C_对话',
                    'chat_conflict', ['显式表达优先'], channel='chat')


def sc_chat_think(r, sp, cid):
    c = r.choice(DUAL_OK)
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    dp = flow(s, [(c, 1)], stage_for(c, r), chat=[(pick(r, CHAT_OTHER['think'], sp), None)])
    return finalize(s, cid, g_act(c, allow_uncertain=True, channel='chat'),
                    make_profile(r, tier=r.choice(['低活跃', '新客'])), dp, 'C_对话', 'chat_think', channel='chat')


def sc_chat_optout(r, sp, cid, optout=True):
    c = r.choice(['价格', '比价', '品质信任'])
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(c, r, it)
    utt = pick(r, CHAT_OTHER['optout' if optout else 'think'], sp)
    dp = flow(s, [(c, 2)], r.choice(['商详加购', '购物车']), chat=[(utt, None)])
    gold = g_no('用户拒绝') if optout else g_act(c, channel='chat')
    return finalize(s, cid, gold, make_profile(r), dp, 'N_不出手' if optout else 'C_对话', 'chat_optout',
                    channel='chat')


def sc_chat_nonshop(r, sp, cid):
    cat, it = pick_item(r, sp)
    s = Session(r, sp, cat, it)
    s.ev.append(s.e('推荐流', s.fmt(pick(r, ENTRY['推荐流'][1], sp))))
    s.ev.append(s.e('商详', f'在商详停留 {r.randint(3, 8)} 秒'))
    s.ev += s.chat([(pick(r, CHAT_OTHER['nonshop'], sp), None)])
    s.ev.append(s.e('商详', pick(r, EXIT['商详'][1], sp)))
    return finalize(s, cid, g_no('非购物意图'), make_profile(r), '商详离开未加购', 'N_不出手', 'chat_nonshop',
                    channel='chat')


def sc_no_mustbuy(r, sp, cid):
    it = copy.deepcopy(r.choice(ITEMS[sp]['快消']))
    s = Session(r, sp, '快消', it)
    k = r.randint(4, 8)
    prof = make_profile(r, '高活跃', '低', [f'该商品复购 {k} 次，平均间隔 {r.randint(30, 50)} 天，距上次购买 {r.randint(28, 48)} 天'])
    s.ev.append(s.e('搜索', s.fmt(pick(r, ['搜索常买的「{name}」', '在「常买清单」点进「{name}」', '从订单历史「再次购买」进入「{name}」'], sp))))
    s.ev.append(s.e('商详', f'停留 {r.randint(3, 8)} 秒'))
    s.ev.append(s.e('商详', '加入购物车'))
    s.ev.append(s.e('结算', '进入结算页'))
    s.ev.append(s.e('结算', f'结算页停留 {r.randint(2, 5)} 秒后退出，订单未支付'))
    return finalize(s, cid, g_no('必买'), prof, '结算页退出', 'N_不出手', 'no_mustbuy')


def sc_no_weak(r, sp, cid):
    cat, it = pick_item(r, sp)
    s = Session(r, sp, cat, it)
    s.ev.append(s.e('推荐流', s.fmt(pick(r, ENTRY[r.choice(['推荐流', '短视频'])][1], sp))))
    s.ev.append(s.e('商详', f'浏览商详 {r.randint(2, 6)} 秒'))
    if r.random() < 0.5:
        s.ev.append(s.lib(NOISE))
    s.ev.append(s.e('商详', pick(r, EXIT['商详'][1], sp)))
    return finalize(s, cid, g_no('弱需求'), make_profile(r), '商详离开未加购', 'N_不出手', 'no_weak')


def sc_no_substitute(r, sp, cid, bought=True):
    cat, it = pick_item(r, sp)
    s = Session(r, sp, cat, it)
    dp = flow(s, [('比价', 2)], '购物车')
    if bought:
        alt = s.base['alt']
        s.ev = s.ev[:-1] + [s.e('结算', f'对「{alt}」下单并支付成功', None, alt),
                            s.e('购物车', f'回到购物车，「{it["name"]}」仍未结算'),
                            s.e('购物车', '离开购物车')]
        return finalize(s, cid, g_no('已购替代', secondary_ok=['比价']), make_profile(r), '购物车离开',
                        'N_不出手', 'no_substitute')
    return finalize(s, cid, g_act('比价'), make_profile(r), dp, 'L1_清晰', 'no_substitute')


def sc_no_freq(r, sp, cid, freq=True):
    c = r.choice(['价格', '比价', '品质信任', '规格适配'])
    cat, it = item_for(r, sp, c)
    s = Session(r, sp, cat, it)
    s.extra = extra_for(c, r, it)
    note = f'近 7 天收到挽留触达 {r.randint(3, 5)} 次，全部点击关闭'
    dp = flow(s, [(c, 2)], stage_for(c, r))
    if freq:
        return finalize(s, cid, g_no('频控', primary=c, score_primary=True), make_profile(r, extras=[note]), dp,
                        'N_不出手', 'no_freq')
    return finalize(s, cid, g_act(c), make_profile(r), dp, 'L1_清晰', 'no_freq')


def sc_no_sensitive(r, sp, cid):
    cat = r.choice(SENSITIVE_CATS)
    it = copy.deepcopy(ITEMS[sp][cat][0])
    s = Session(r, sp, cat, it)
    dp = flow(s, [('价格', 2)], '购物车')
    return finalize(s, cid, g_no('敏感类目'), make_profile(r), dp, 'N_不出手', 'no_sensitive')


def sc_no_hoarder(r, sp, cid):
    cat, it = pick_item(r, sp)
    s = Session(r, sp, cat, it)
    n, x = r.randint(30, 80), r.randint(40, 120)
    prof = make_profile(r, extras=[f'购物车长期有 {n} 件商品，近 90 天加购 {x} 次、只结算 {r.randint(1, 3)} 次'])
    dp = flow(s, [], '商详加购', neutral_k=1)
    return finalize(s, cid, g_no('囤货型'), prof, dp, 'N_不出手', 'no_hoarder')


def sc_trap_profile_conflict(r, sp, cid):
    cat, it = item_for(r, sp, '规格适配')
    s = Session(r, sp, cat, it)
    prof = make_profile(r, price_sens='高', extras=['近 30 天领券 6 次、核销 4 次'])
    dp = flow(s, [('规格适配', 3)], stage_for('规格适配', r))
    return finalize(s, cid, g_act('规格适配', must_not=['价格']), prof, dp, 'L5_陷阱', 'trap_profile_conflict')


def sc_trap_lowest_price(r, sp, cid, lowest=True):
    cat, it = item_for(r, sp, '价格')
    if lowest:
        it['price'] = it['low']
    s = Session(r, sp, cat, it)
    dp = flow(s, [('价格', 3)], stage_for('价格', r))
    gold = g_act('价格', action_ok=['价格事实'], must_not=['定向券']) if lowest else g_act('价格')
    return finalize(s, cid, gold, make_profile(r, price_sens='高'), dp, 'L5_陷阱' if lowest else 'L1_清晰',
                    'trap_lowest_price')


def sc_trap_coupon_waiter(r, sp, cid):
    cat, it = item_for(r, sp, '价格')
    s = Session(r, sp, cat, it)
    prof = make_profile(r, price_sens='高', extras=[f'近 90 天有 {r.randint(4, 7)} 次「加购后收到挽留券才下单」'])
    dp = flow(s, [('价格', 2)], '购物车')
    return finalize(s, cid, g_act('价格', action_ok=['价格事实', '价保'], must_not=['定向券']), prof, dp,
                    'L5_陷阱', 'trap_coupon_waiter')


def sc_trap_focus_item(r, sp, cid):
    cat, it = item_for(r, sp, '规格适配')
    s = Session(r, sp, cat, it)
    other = pick_item(r, sp, cats=[cat])[1]
    while other['name'] == it['name']:
        other = pick_item(r, sp)[1]
    dp = flow(s, [('规格适配', 2)], stage_for('规格适配', r), others=[(other, '价格', 2)])
    return finalize(s, cid, g_act('规格适配', must_not=['价格']), make_profile(r), dp, 'L5_陷阱', 'trap_focus_item')


def sc_trap_live_expiring(r, sp, cid):
    cat, it = item_for(r, sp, '价格')
    s = Session(r, sp, cat, it)
    s.extra = [f'直播间专享价比日常价低 ¥{max(5, round(it["price"] * 0.08))}，今晚 24 点失效']
    dp = flow(s, [('价格', 2)], r.choice(['商详加购', '购物车']), entry='直播间')
    gold = g_act('价格', action_ok=['价格事实'], must_not=['定向券'], touch_ok=['push'])
    return finalize(s, cid, gold, make_profile(r), dp, 'L5_陷阱', 'trap_live_expiring')


def sc_trap_installment(r, sp, cid):
    cat, it = pick_item(r, sp, min_price=2000)
    s = Session(r, sp, cat, it)
    inst = s.e('商详', f'查看分期付款方案（{r.choice([3, 6, 12])} 期免息），停留 {r.randint(15, 50)} 秒', '价格')
    dp = flow(s, [('价格', 1)], r.choice(['商详', '购物车']), first_events=[])
    s.ev.insert(len(s.ev) - 1, inst)
    gold = g_act('价格', action_ok=['分期信息', '价格事实'])
    return finalize(s, cid, gold, make_profile(r, price_sens=r.choice(['中', '高'])), dp, 'L5_陷阱', 'trap_installment')


def sc_trap_newbie_trust(r, sp, cid):
    cat, it = item_for(r, sp, '品质信任')
    s = Session(r, sp, cat, it)
    s.extra = [f'店铺开店 {r.randint(2, 6)} 个月，店铺评分 4.2']
    prof = make_profile(r, tier='新客', extras=['平台首单未完成'])
    dp = flow(s, [('品质信任', r.randint(2, 3))], stage_for('品质信任', r))
    return finalize(s, cid, g_act('品质信任'), prof, dp, 'L5_陷阱', 'trap_newbie_trust')


def sc_trap_gift(r, sp, cid):
    cat, it = item_for(r, sp, '规格适配')
    s = Session(r, sp, cat, it)
    gift = s.e('搜索', s.fmt(pick(r, ['搜索「送男朋友 {name}」', '搜索「送妈妈 {name}」', '搜索「{name} 礼物」'], sp)))
    dp = flow(s, [('规格适配', 2)], stage_for('规格适配', r), entry='搜索', first_events=[gift])
    return finalize(s, cid, g_act('规格适配'), make_profile(r), dp, 'L5_陷阱', 'trap_gift', ['送礼'])


BUILDERS = {
    'app_clear': sc_app_clear, 'app_noisy': sc_app_noisy, 'app_sparse': sc_app_sparse, 'app_dual': sc_app_dual,
    'chat_explicit': sc_chat_explicit, 'chat_implicit': sc_chat_implicit, 'chat_conflict': sc_chat_conflict,
    'chat_think': sc_chat_think, 'chat_optout': sc_chat_optout, 'chat_nonshop': sc_chat_nonshop,
    'no_mustbuy': sc_no_mustbuy, 'no_weak': sc_no_weak, 'no_substitute': sc_no_substitute, 'no_freq': sc_no_freq,
    'no_sensitive': sc_no_sensitive, 'no_hoarder': sc_no_hoarder,
    'trap_profile_conflict': sc_trap_profile_conflict, 'trap_lowest_price': sc_trap_lowest_price,
    'trap_coupon_waiter': sc_trap_coupon_waiter, 'trap_focus_item': sc_trap_focus_item,
    'trap_live_expiring': sc_trap_live_expiring, 'trap_installment': sc_trap_installment,
    'trap_newbie_trust': sc_trap_newbie_trust, 'trap_gift': sc_trap_gift,
}

DEFAULT_TEST_PLAN = {}
for _c in CONCERNS:
    DEFAULT_TEST_PLAN.update({f'app_clear:{_c}': 4, f'app_noisy:{_c}': 6, f'app_sparse:{_c}': 2,
                              f'chat_explicit:{_c}': 3, f'chat_implicit:{_c}': 2})
DEFAULT_TEST_PLAN.update({
    'app_dual': 30, 'chat_conflict': 10, 'chat_think': 6, 'chat_optout': 8, 'chat_nonshop': 8,
    'no_mustbuy': 14, 'no_weak': 14, 'no_substitute': 10, 'no_freq': 10, 'no_sensitive': 8, 'no_hoarder': 8,
    'trap_profile_conflict': 10, 'trap_lowest_price': 8, 'trap_coupon_waiter': 8, 'trap_focus_item': 8,
    'trap_live_expiring': 6, 'trap_installment': 6, 'trap_newbie_trust': 6, 'trap_gift': 6,
})

# 最小对比对：同一段会话只改一个条件，答案就该翻转
PAIRS = {
    'pair_lowest': ('trap_lowest_price', 'lowest'),
    'pair_freq': ('no_freq', 'freq'),
    'pair_substitute': ('no_substitute', 'bought'),
    'pair_optout': ('chat_optout', 'optout'),
}


def build(key, r, split, cid, **kw):
    name, _, arg = key.partition(':')
    fn = BUILDERS[name]
    return fn(r, split, cid, arg, **kw) if arg else fn(r, split, cid, **kw)


def generate(split, seed, plan, prefix=None):
    r = random.Random(seed)
    prefix = prefix or ('T' if split == 'test' else 'R')
    cases = []
    for key, n in plan.items():
        for _ in range(n):
            cases.append(build(key, r, split, f'{prefix}{len(cases) + 1:04d}'))
    return cases


def generate_pairs(split, seed, n_per_pair, prefix='P'):
    cases, k = [], 0
    base = random.Random(seed)
    for pname, (scen, flag) in PAIRS.items():
        for _ in range(n_per_pair):
            s = base.randrange(1 << 30)
            k += 1
            a = build(scen, random.Random(s), split, f'{prefix}{k:04d}a', **{flag: True})
            b = build(scen, random.Random(s), split, f'{prefix}{k:04d}b', **{flag: False})
            for c in (a, b):
                c['tags'] = c['tags'] + [pname]
            cases += [a, b]
    return cases


def render(case, ablate=()):
    lines = [f'【渠道】{"App 行为 + 客服对话" if case["channel"] == "chat" else "App 行为"}', '']
    if 'profile' not in ablate:
        p = case['profile']
        s = f'活跃度：{p["tier"]}；价格敏感度：{p["price_sens"]}'
        if p['extras']:
            s += '；' + '；'.join(p['extras'])
        lines += ['【用户画像】', s, '']
    if 'item' not in ablate:
        it = case['item']
        s = (f'{it["name"]}（{it["category"]}），当前价 ¥{fmt_price(it["price"])}，近 30 天均价 ¥{fmt_price(it["avg"])}，'
             f'近 30 天最低 ¥{fmt_price(it["low"])}；评分 {it["rating"]}；{it["ship"]}')
        if it['extra']:
            s += '；' + '；'.join(it['extra'])
        lines += ['【商品上下文】', s, '']
    lines.append(f'【会话行为】截至决策时刻，共 {len(case["events"])} 条')
    for e in case['events']:
        tag = f'{e["page"]}·{e["item"]}' if case['multi_item'] and e.get('item') else e['page']
        lines.append(f'#{e["i"]} {e["ts"]} [{tag}] {e["desc"]}')
    lines += ['', f'【决策时刻】#{len(case["events"])} {case["decision_point"]}（焦点商品：{case["item"]["name"]}）']
    return '\n'.join(lines)


def gold_answer(case):
    """把标准答案写成模型应输出的 JSON，用于 SFT。"""
    g = case['gold']
    ev = {e['i']: e['desc'] for e in case['events']}
    if g['act']:
        reason = '依据 ' + '；'.join(f'#{i} {ev[i]}' for i in g['evidence'][:3]) if g['evidence'] else '证据较少'
    else:
        reason = NO_ACT_TEXT[g['act_reason']]
    return dict(act=g['act'], primary=g['primary'], secondary=(g['secondary_ok'] or [None])[0],
                confidence=0.5 if g['allow_uncertain'] else 0.85, evidence=g['evidence'],
                action=g['action_ok'][0], touchpoint=g['touch_ok'][0], uncertain=g['allow_uncertain'],
                reason=reason)
