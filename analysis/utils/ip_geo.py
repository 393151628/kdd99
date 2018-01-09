# -*- coding: utf-8 -*-
import urllib.request as urllib2
import re
from random import randint

geo_map = {
    '东萨摩亚(美)': 'Samoa Eastern',
    '中国': 'China',
    '中非共和国': 'Central African Republic',
    '丹麦': 'Denmark',
    '乌克兰': 'Ukraine',
    '乌兹别克斯坦': 'Uzbekistan',
    '乌干达': 'Uganda',
    '乌拉圭': 'Uruguay',
    '乍得': 'Chad',
    '也门': 'Yemen',
    '亚美尼亚': 'Armenia',
    '以色列': 'Israel',
    '伊拉克': 'Iraq',
    '伊朗': 'Iran',
    '伯利兹': 'Belize',
    '俄罗斯': 'Russia',
    '保加利亚': 'Bulgaria',
    '关岛': 'Guam',
    '冈比亚': 'Gambia',
    '冰岛': 'Iceland',
    '几内亚': 'Guinea',
    '列支敦士登': 'Liechtenstein',
    '刚果': 'Congo',
    '利比亚': 'Libya',
    '利比里亚': 'Liberia',
    '加拿大': 'Canada',
    '加纳': 'Ghana',
    '加蓬': 'Gabon',
    '匈牙利': 'Hungary',
    '南斯拉夫': 'Yugoslavia',
    '南非': 'South Africa',
    '博茨瓦纳': 'Botswana',
    '卡塔尔': 'Qatar',
    '卢森堡': 'Luxembourg',
    '印度': 'India',
    '印度尼西亚': 'Indonesia',
    '危地马拉': 'Guatemala',
    '厄瓜多尔': 'Ecuador',
    '叙利亚': 'Syria',
    '古巴': 'Cuba',
    '台湾省': 'China',
    '吉尔吉斯坦': 'Kyrgyzstan',
    '吉布提': 'Djibouti',
    '哈萨克斯坦': 'Kazakstan',
    '哥伦比亚': 'Colombia',
    '哥斯达黎加': 'Costa Rica',
    '喀麦隆': 'Cameroon',
    '土库曼斯坦': 'Turkmenistan',
    '土耳其': 'Turkey',
    '圣卢西亚': 'St.Lucia',
    '圣多美和普林西比': 'Sao Tome and Principe',
    '圣文森特': 'St.Vincent',
    '圣文森特岛': 'Saint Vincent',
    '圣马力诺': 'San Marino',
    '圭亚那': 'Guyana',
    '坦桑尼亚': 'Tanzania',
    '埃及': 'Egypt',
    '埃塞俄比亚': 'Ethiopia',
    '塔吉克斯坦': 'Tajikstan',
    '塞内加尔': 'Senegal',
    '塞拉利昂': 'Sierra Leone',
    '塞浦路斯': 'Cyprus',
    '塞舌尔': 'Seychelles',
    '墨西哥': 'Mexico',
    '多哥': 'Togo',
    '多米尼加共和国': 'Dominica Rep.',
    '奥地利': 'Austria',
    '委内瑞拉': 'Venezuela',
    '孟加拉国': 'Bangladesh',
    '安圭拉岛': 'Anguilla',
    '安提瓜和巴布达': 'Antigua and Barbuda',
    '安道尔共和国': 'Andorra',
    '尼加拉瓜': 'Nicaragua',
    '尼日利亚': 'Nigeria',
    '尼日尔': 'Niger',
    '尼泊尔': 'Nepal',
    '巴哈马': 'Bahamas',
    '巴基斯坦': 'Pakistan',
    '巴巴多斯': 'Barbados',
    '巴布亚新几内亚': 'Papua New Cuinea',
    '巴拉圭': 'Paraguay',
    '巴拿马': 'Panama',
    '巴林': 'Bahrain',
    '巴西': 'Brazil',
    '布基纳法索': 'Burkina-faso',
    '布隆迪': 'Burundi',
    '希腊': 'Greece',
    '库克群岛': 'Cook Is.',
    '开曼群岛': 'Cayman Is.',
    '德国': 'Germany',
    '意大利': 'Italy',
    '所罗门群岛': 'Solomon Is',
    '扎伊尔': 'Zaire',
    '拉脱维亚': 'Latvia',
    '挪威': 'Norway',
    '捷克': 'Czech Republic',
    '摩尔多瓦': 'Moldova, Republic of',
    '摩洛哥': 'Morocco',
    '摩纳哥': 'Monaco',
    '文莱': 'Brunei',
    '斐济': 'Fiji',
    '斯威士兰': 'Swaziland',
    '斯洛伐克': 'Slovakia',
    '斯洛文尼亚': 'Slovenia',
    '斯里兰卡': 'Sri Lanka',
    '新加坡': 'Singapore',
    '新西兰': 'New Zealand',
    '日本': 'Japan',
    '智利': 'Chile',
    '朝鲜': 'North Korea',
    '柬埔寨': 'Kampuchea (Cambodia )',
    '格林纳达': 'Grenada',
    '格鲁吉亚': 'Georgia',
    '比利时': 'Belgium',
    '毛里求斯': 'Mauritius',
    '汤加': 'Tonga',
    '沙特阿拉伯': 'Saudi Arabia',
    '法国': 'France',
    '法属圭亚那': 'French Guiana',
    '法属玻利尼西亚': 'French Polynesia',
    '波兰': 'Poland',
    '波多黎各': 'Puerto Rico',
    '泰国': 'Thailand',
    '津巴布韦': 'Zimbabwe',
    '洪都拉斯': 'Honduras',
    '海地': 'Haiti',
    '澳大利亚': 'Australia',
    '澳门': 'Macao',
    '爱尔兰': 'Ireland',
    '爱沙尼亚': 'Estonia',
    '牙买加': 'Jamaica',
    '特立尼达和多巴哥': 'Trinidad and Tobago',
    '玻利维亚': 'Bolivia',
    '瑙鲁': 'Nauru',
    '瑞典': 'Sweden',
    '瑞士': 'Switzerland',
    '留尼旺': 'Reunion',
    '白俄罗斯': 'Belarus',
    '百慕大群岛': 'Bermuda Is.',
    '直布罗陀': 'Gibraltar',
    '科威特': 'Kuwait',
    '科特迪瓦': 'Ivory Coast',
    '秘鲁': 'Peru',
    '突尼斯': 'Tunisia',
    '立陶宛': 'Lithuania',
    '索马里': 'Somali',
    '约旦': 'Jordan',
    '纳米比亚': 'Namibia',
    '缅甸': 'Burma',
    '罗马尼亚': 'Romania',
    '美国': 'United States of America',
    '老挝': 'Laos',
    '肯尼亚': 'Kenya',
    '芬兰': 'Finland',
    '苏丹': 'Sudan',
    '苏里南': 'Suriname',
    '英国': 'United Kiongdom',
    '荷兰': 'Netherlands',
    '荷属安的列斯': 'Netheriands Antilles',
    '莫桑比克': 'Mozambique',
    '莱索托': 'Lesotho',
    '菲律宾': 'Philippines',
    '萨尔瓦多': 'EI Salvador',
    '葡萄牙': 'Portugal',
    '蒙古': 'Mongolia',
    '蒙特塞拉特岛': 'Montserrat Is',
    '西班牙': 'Spain',
    '西萨摩亚': 'Samoa Western',
    '贝宁': 'Benin',
    '赞比亚': 'Zambia',
    '越南': 'Vietnam',
    '阿塞拜疆': 'Azerbaijan',
    '阿富汗': 'Afghanistan',
    '阿尔及利亚': 'Algeria',
    '阿尔巴尼亚': 'Albania',
    '阿拉伯联合酋长国': 'United Arab Emirates',
    '阿曼': 'Oman',
    '阿根廷': 'Argentina',
    '阿森松': 'Ascension',
    '韩国': 'Korea',
    '香港': 'Hongkong',
    '马尔代夫': 'Maldives',
    '马拉维': 'Malawi',
    '马提尼克': 'Martinique',
    '马来西亚': 'Malaysia',
    '马耳他': 'Malta',
    '马达加斯加': 'Madagascar',
    '马里': 'Mali',
    '马里亚那群岛': 'Mariana Is',
    '黎巴嫩': 'Lebanon',
    '本地局域网':'LAN',
    '保留地址': 'Reserved Address'
}

geo_names = [
    "China","Somalia","Liechtenstein","Morocco","W. Sahara","Serbia","Afghanistan","Angola",
    "Albania","Aland","Andorra","United Arab Emirates","Argentina","Armenia","American Samoa",
    "Fr. S. Antarctic Lands","Antigua and Barb.","Australia","Austria","Azerbaijan","Burundi",
    "Belgium","Benin","Burkina Faso","Bangladesh","Bulgaria","Bahrain","Bahamas",
    "Bosnia and Herz.","Belarus","Belize","Bermuda","Bolivia","Brazil","Barbados","Brunei",
    "Bhutan","Botswana","Central African Rep.","Canada","Switzerland","Chile","Côte d'Ivoire",
    "Cameroon","Dem. Rep. Congo","Congo","Colombia","Comoros","Cape Verde","Costa Rica",
    "Cuba","Curaçao","Cayman Is.","N. Cyprus","Cyprus","Czech Rep.","Germany","Djibouti",
    "Dominica","Denmark","Dominican Rep.","Algeria","Ecuador","Egypt","Eritrea","Spain",
    "Estonia","Ethiopia","Finland","Fiji","Falkland Is.","France","Faeroe Is.",
    "Micronesia","Gabon","United Kingdom","Georgia","Ghana","Guinea","Gambia",
    "Guinea-Bissau","Eq. Guinea","Greece","Grenada","Greenland","Guatemala","Guam",
    "Guyana","Heard I. and McDonald Is.","Honduras","Croatia","Haiti","Hungary",
    "Indonesia","Isle of Man","India","Br. Indian Ocean Ter.","Ireland","Iran","Iraq",
    "Iceland","Israel","Italy","Jamaica","Jersey","Jordan","Japan","Siachen Glacier",
    "Kazakhstan","Kenya","Kyrgyzstan","Cambodia","Kiribati","Korea","Kuwait","Lao PDR",
    "Lebanon","Liberia","Libya","Saint Lucia","Sri Lanka","Lesotho","Lithuania",
    "Luxembourg","Latvia","Moldova","Madagascar","Mexico","Macedonia","Mali","Malta",
    "Myanmar","Montenegro","Mongolia","N. Mariana Is.","Mozambique","Mauritania",
    "Montserrat","Mauritius","Malawi","Malaysia","Namibia","New Caledonia","Niger",
    "Nigeria","Nicaragua","Niue","Netherlands","Norway","Nepal","New Zealand","Oman",
    "Pakistan","Panama","Peru","Philippines","Palau","Papua New Guinea","Poland",
    "Puerto Rico","Dem. Rep. Korea","Portugal","Paraguay","Palestine","Fr. Polynesia",
    "Qatar","Romania","Russia","Rwanda","Saudi Arabia","Sudan","S. Sudan","Senegal",
    "Singapore","S. Geo. and S. Sandw. Is.","Saint Helena","Solomon Is.","Sierra Leone",
    "El Salvador","St. Pierre and Miquelon","São Tomé and Principe","Suriname","Slovakia",
    "Slovenia","Sweden","Swaziland","Seychelles","Syria","Turks and Caicos Is.","Chad",
    "Togo","Thailand","Tajikistan","Turkmenistan","Timor-Leste","Tonga",
    "Trinidad and Tobago","Tunisia","Turkey","Tanzania","Uganda","Ukraine","Uruguay",
    "United States","Uzbekistan","St. Vin. and Gren.","Venezuela","U.S. Virgin Is.",
    "Vietnam","Vanuatu","Samoa","Yemen","South Africa","Zambia","Zimbabwe"]


def get_geo_name_by_ip(ip_addr):
    # ipaddr = '80.209.231.191'
    # url = "http://www.ip138.com/ips138.asp?ip=%s&action=2" % ip_addr
    # print(url)
    # u = urllib2.urlopen(url)
    # s = u.read()
    # # Get IP Address
    # s = str(s, encoding='gbk')
    # # Get IP Address Location
    # result = re.findall(r'(<li>.*?</li>)', s)[0]
    # country = result[9:-5].strip()
    # return geo_map.get(country, 'unknow')

    return geo_names[randint(0,200)]


if __name__ == '__main__':
    ipaddr = '80.209.231.191'
    print(get_geo_name_by_ip(ipaddr))


