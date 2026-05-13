# 高频量化血泪史 - 停更！但是分享一下小技巧

这一篇是最后的部分了。分享一个和北面老师聊完后的工程学tricks。 这里做的是单交易所双腿（不做跨所对冲）的做市版本。 先说跨所对冲的优缺点： 优势：delta中性，灵活。收益稳定。可以做双边maker的XEMM，也可以做一边maker一边taker的Arbitrage模式。策略容量大，能跑大币对，9-8 figures available 缺点：套利机会越来越机会少，竞争激烈。同 alpha- 价差 用的人越来越多。每次下单量太大，有时不一定单腿总fill，形成头寸暴露。两所仓位保证金平衡起来麻烦，总涉及互相转账的问题。 我直接一个头铁拥抱尾部风险搞单所双腿了。 下面说tricks 1. 框架上，我们团队花了将近一个月，把线性的计算改成共享内存的微服务了。这样绑核了算数据，算完了以后共享内存所有策略都可以调用。 2. 服务器做master - slave node rpc控制，所有子账号绑定elastic ip。用aws eni 开单独接口分行情，一个接colo听 一个eni开多elastic 剩下的用普通ip下。每个eip绑一组api，绑定到多子账户多实例。这里注意，eip不够的话需要和aws单独申请他们的quoting service 3 由于单个ticker/symbol盈亏不可控、面板调用pid，每个小资金100-200 分开跑不同ticker。哪个ticker突然在滚动窗口内突然盈利很多，就集中跑这个ticker，直到cap 打满。 这样做的原因是：思考一下策略有挂单和撤单-当定价决定撤改单，说明这里是个坏价。next price move是个fresh price。如果多个sub打一个ticker，总有instance能挂到 fresh price靠前的队列，平均了网络jitter和队列加入成本。 然后说一个头铁做单所的库存思路。 首先持仓有风险这个大家是公认的。持仓时间越长theata越大 但是但是但是。我们实际是可以控制库存区间的。我的做法是让持仓库存小于每次的order qty，这样的好处就是仓位每次cross 0，正负号变了，我就可以认为这比头寸结算了，进入下一个循环。库存大于一个order qty 就强skew清掉，直到只剩一个小于order qty的小库存。 篇幅有限，写的比较不全面。很多没解释到位的地方大家用ai看一下吧～

# 高频做市最终章- 自查表

![[Pasted image 20260412065742.png]]

最近加了很多xhs的朋友一起交流。文本长度有限。接下来我把我的整个开发流程+策略流程以最简单粗暴的关键词和benchmark形式列表，记录了我做hftmm以来既涵盖的所有坑，方便友友们自查。不会的直接问ai 1 aws东京ap1a-az4 ec2 20个instance一组。c7i/c8i xlarge至少（不要用小土豆）行情benchmark蒸馏可实现最小单位scp到服务器收data。future p50 local ts - T 1.3-1.8ms codec零拷贝解析json。spot SBE+ Fix 400us-900us。ipc回传，本地维护ob depth@0ms ringbuffer管理。双轨订阅。 2 ws下单 rest batch order保护api token 如微服务用令牌桶。执行p50小于30us + 行情延迟就不下单的风控。kill switch风控单独管理mdd 3回测做跨线算fill，队列模型+延迟模型确保和实盘框架align 数据用本所，通用精度可以用tardisdev 4策略层直接用greeks管理。w1 *alpha +- w2* beta….做报价偏移。轮询或时序看策略风格。只管理greeks种类就行了。ob算alpha，下单下bbo。event2order 3.2ms-3.8ms 。小于5ms算盈利区 colo wl 不改变速度只改变大行情通道堵不堵。alpha beta delta gamma theta…..每个导入需要被管理的公式和函数。多级下单平均成本价。动态库存管理简化as就够。不需要预测return，或者-1/0/1归一方向 以上，图一是不使用ml任何模型（不吃黑盒收益，手续费不敏感，纯解释性规则的gtx策略：纯greeks加权）的回测数据。图二实盘。 波动率用iv的思路。 有问题评论区留言，本人不是大佬，但一定知无不言，多个朋友多条路，peace![](https://picasso-static.xiaohongshu.com/fe-platform/d4fe00be555964ddf8301e256cd906b9032679a5.png)

03-23 日本

共 75 条评论

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31ubmq74ihm004bmb5mep7vaj9d0rj98?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5bd85d93cd7a4a00016efd53?channel_type=web_profile_page&xsec_token=ABIIOCd_2Q9Tlol19KR6j-lKE9JeU-l9mKzSjRav4COrY%3D&xsec_source=pc_comment)

[double](https://www.xiaohongshu.com/user/profile/5bd85d93cd7a4a00016efd53?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABIIOCd_2Q9Tlol19KR6j-lKE9JeU-l9mKzSjRav4COrY=&xsec_source=pc_comment)

双轨订阅是指什么

03-23江苏

赞

1

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

两条ws行情维护本地

03-23日本

2

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31n5eb5rf5i0g5orsgie7qkv2dfv8ato?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/637c849c000000001f0153e2?channel_type=web_profile_page&xsec_token=ABU6AFofyh_nqm6ysOHJXKiSCWq2XCJdG2Rtm7SxJXNjI%3D&xsec_source=pc_comment)

[momooooo](https://www.xiaohongshu.com/user/profile/637c849c000000001f0153e2?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABU6AFofyh_nqm6ysOHJXKiSCWq2XCJdG2Rtm7SxJXNjI=&xsec_source=pc_comment)

请问大佬说的用SBE 维护depth@0ms 是什么意思？ SBE diff depth 更新不是50ms吗？是还有其他办法获得实时更新的的OB diff 吗![](https://picasso-static.xiaohongshu.com/fe-platform/9366d16631e3e208689cbc95eefb7cfb0901001e.png)，多谢🙏

03-27瑞士

赞

1

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

depth 0ms 是future的 ws写法incremental，sbe我只做了spot的bbo

03-29中国香港

1

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31g74fuerhe6g5p98imcal4n6lfe7ogo?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/65289598000000002a0292e6?channel_type=web_profile_page&xsec_token=ABbCSN4KebaFRd9pklgU8-dzvGxvjrzDNSsg7gQbizTLc%3D&xsec_source=pc_comment)

[41ik](https://www.xiaohongshu.com/user/profile/65289598000000002a0292e6?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABbCSN4KebaFRd9pklgU8-dzvGxvjrzDNSsg7gQbizTLc=&xsec_source=pc_comment)

请问depth@0ms维护全book还是裁切到需要的范围

03-23北京

赞

3

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

增量

03-23日本

1

回复

展开 2 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31n5eb5rf5i0g5orsgie7qkv2dfv8ato?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/637c849c000000001f0153e2?channel_type=web_profile_page&xsec_token=ABU6AFofyh_nqm6ysOHJXKiSCWq2XCJdG2Rtm7SxJXNjI%3D&xsec_source=pc_comment)

[momooooo](https://www.xiaohongshu.com/user/profile/637c849c000000001f0153e2?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABU6AFofyh_nqm6ysOHJXKiSCWq2XCJdG2Rtm7SxJXNjI=&xsec_source=pc_comment)

请问那个线性预测是预测的什么值？多久之后的呢？

03-23美国

赞

4

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

没有做预测，没有做ml

03-24北京

赞

回复

展开 3 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/645654a2c2729fae1a9cff59.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5ae49c9f4eacab3dc143bb91?channel_type=web_profile_page&xsec_token=AB5CdT40zewdpbqQF9em2l_FVH5k8isCEXxHs2x7vH2Pg%3D&xsec_source=pc_comment)

[hhhcc](https://www.xiaohongshu.com/user/profile/5ae49c9f4eacab3dc143bb91?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=AB5CdT40zewdpbqQF9em2l_FVH5k8isCEXxHs2x7vH2Pg=&xsec_source=pc_comment)

请问从哪里可以看到您的实盘

03-23江苏

1

1

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

不给看不对外嘻嘻

03-23日本

赞

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/6590dbc6d4b53385b6a6bd1c.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5f6b6f040000000001004186?channel_type=web_profile_page&xsec_token=AB4wgazV5f80NIRcM68sZdYWqkIqweoSame78SOGW-BKA%3D&xsec_source=pc_comment)

[winwin](https://www.xiaohongshu.com/user/profile/5f6b6f040000000001004186?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=AB4wgazV5f80NIRcM68sZdYWqkIqweoSame78SOGW-BKA=&xsec_source=pc_comment)

太厉害了 下单bbo什么意思 所有下单都在best bid ask之间吗

03-24美国

赞

3

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

bbo行情快 所以你的报价应该围绕bbo行情+- greeks 不要用top of orderbook的行情

03-25北京

1

回复

展开 2 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/5b1beba5f7e8b9240752dbda.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5b1beba5f7e8b9240752dbda?channel_type=web_profile_page&xsec_token=ABF9lSi3sgb657ZljPJs5WYBwVio0XhTCwlkiTE1Y11Co%3D&xsec_source=pc_comment)

[qijun.L](https://www.xiaohongshu.com/user/profile/5b1beba5f7e8b9240752dbda?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABF9lSi3sgb657ZljPJs5WYBwVio0XhTCwlkiTE1Y11Co=&xsec_source=pc_comment)

只交易币吗？

03-25广东

赞

2

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

先起步，再扩市场

03-25北京

赞

回复

展开 1 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31stc6si6lm004a7n06nsomei2nrp7i0?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5b5fefcc11be104b556a59d2?channel_type=web_profile_page&xsec_token=ABr4MIeouszWHXhVshUVgR55bOJayqyNdK0GJHfdS-3II%3D&xsec_source=pc_comment)

[堵怪](https://www.xiaohongshu.com/user/profile/5b5fefcc11be104b556a59d2?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABr4MIeouszWHXhVshUVgR55bOJayqyNdK0GJHfdS-3II=&xsec_source=pc_comment)

太强了bro![](https://picasso-static.xiaohongshu.com/fe-platform/14b005f7afd5f7c88620478b610bf1de90c4ceab.png)![](https://picasso-static.xiaohongshu.com/fe-platform/14b005f7afd5f7c88620478b610bf1de90c4ceab.png)

03-23江苏

赞

1

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

幸福来得太突然

03-23北京

赞

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/645b7ecfb46568f58064957a.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/65bdf3ff000000000e0250c1?channel_type=web_profile_page&xsec_token=ABYMRJiXvM_KXZnL6NayYmE-rY8ZSoOQcjNF-bxDcM558%3D&xsec_source=pc_comment)

[小红薯65BE03BF](https://www.xiaohongshu.com/user/profile/65bdf3ff000000000e0250c1?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABYMRJiXvM_KXZnL6NayYmE-rY8ZSoOQcjNF-bxDcM558=&xsec_source=pc_comment)

每日20w撤单限制怎么解决的？

03-26中国台湾

赞

1

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

用波动率更新同步在循环外，改单的逻辑做。不是撤+挂新的，而是如果变价或者坏价多少个时间窗口ticks改单。持续报价打不到撤单限制的。撤单限制防的是spoofing

03-26中国香港

赞

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31jt9vea33q6g5nddn2t08in7tg94lh0?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5dadb8ba0000000001004ae7?channel_type=web_profile_page&xsec_token=ABhTRk44muJeWw5Wr2JBGbQZsGj0dQsqJKv2e_fhneQ0A%3D&xsec_source=pc_comment)

[自研自用可交流不合作](https://www.xiaohongshu.com/user/profile/5dadb8ba0000000001004ae7?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABhTRk44muJeWw5Wr2JBGbQZsGj0dQsqJKv2e_fhneQ0A=&xsec_source=pc_comment)

请问绿色229%那个是什么数据？

03-26广东

赞

4

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

等比年化

03-26天津

赞

回复

展开 3 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/635752f5b8d155d0e01ee2c8.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5a05d79cb1da1415ccadc4db?channel_type=web_profile_page&xsec_token=AB8FfoRi_Fin25bUpH3NYWf4ZCCSZnTLi1xvnivJ8BLwA%3D&xsec_source=pc_comment)

[好好睡觉呢](https://www.xiaohongshu.com/user/profile/5a05d79cb1da1415ccadc4db?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=AB8FfoRi_Fin25bUpH3NYWf4ZCCSZnTLi1xvnivJ8BLwA=&xsec_source=pc_comment)

请问这是纯币为什么会有greeks呀？还是说的risk factor呀？

03-30美国

赞

2

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo311p7alud0g705nuujligbnm7jiq13qg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5fde9d65000000000101dec7?channel_type=web_profile_page&xsec_token=ABfRKLs_yXdlZYtS7t4jfsswzKoL5m51urfey3XlWyzhk%3D&xsec_source=pc_comment)

[我爱工作废寝忘食](https://www.xiaohongshu.com/user/profile/5fde9d65000000000101dec7?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABfRKLs_yXdlZYtS7t4jfsswzKoL5m51urfey3XlWyzhk=&xsec_source=pc_comment)

应该是feature或者是factor

# 走窄门，烧冷灶。从相亲的角度展开讲讲量化

![[Pasted image 20260412071153.png]]
![[Pasted image 20260412071225.png]]
![[Pasted image 20260412071241.png]]

最近抖音老是给我推荐一些月老，相亲类连麦的，看用户画像的。特别有意思的一句话就是。怎么用你的长板去打别人的短板。走窄门，烧冷灶。精确匹配。量化里的alpha也是。你年龄26，貌美如花，这是alpha，对方离异带娃，这是定价错误。你看的是他四十亿的身家，结婚嫁给金钱而不是爱情，是你的报价。 做市策略这么难，为什么不从简单一些的套利，资费，规则性的策略入手？因为做的人少，而且做市商三个字听起来很帅。就是因为做的少，所以对于家道中落的不思进取的中产小康阶层才有翻身实现财富自由的机会。说白了，是为了赚钱，也是为了自我实现。 知乎是有效市场，大家分享知识，分享alpha，分享规则论文，数学公式函数。 小红书是生活，大家分享穿搭，分享去哪儿吃，去哪儿玩。分享化妆造型，影视解说。 在xhs分享量化的原因是为了在这个用户群体里施展我自己的alpha，一个人设，能逆风翻盘的艺术生，在无效市场靠着完全短板打长板的路子进行差异化竞争：放心 后面肯定去卖课。从零开始，谁都能学，能做量化，能实现财富自由的人设。 alpha会decay，核心可解释性的原理逻辑会渐渐失效。而这短短仅有的一行代码，立刻让我的策略扭亏为盈。我没对任何人提过这个算法。而我，忏悔中，发下一篇又一篇的帖子，在用这种狡猾的市场方式对冲我自己的beta。放心，我真的会去卖课。 我从小接受的教育是充满了极端精英主义的，个人英雄主义，自我中心的意识形态高度内在化。所以我可以开诚布公地吐露心声。告诉大家我的开发流程。我受益于教育，也将成为知识布道的拥护者。自恋主义作祟下，我想撑伞，所有人都可以走伞下经过。希望我的学习量化的思路，开发流程能帮到更多人。卖课只需要录制视频的成本，不需要扩张边界。只需要合眼缘，就像相亲一样。 统计一下会有多少人对高频做市+低延迟的课程感兴趣。虽然我不够专业，但是能给大家分享一下经验和工作流。 顺便赚一点钱。![](https://picasso-static.xiaohongshu.com/fe-platform/d1a34cf8aeac526d36890d3e8f727192a6808ecf.png)

03-27 中国香港

共 8 条评论

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31uhvjmokia6g5og1cqb41kqfrbn8r40?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/62016696000000001000d34f?channel_type=web_profile_page&xsec_token=ABkHf-59xOfzLXZQqC0a8MHJTL5kNTshAKl9kJqOvFYaI%3D&xsec_source=pc_comment)

[安河桥](https://www.xiaohongshu.com/user/profile/62016696000000001000d34f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABkHf-59xOfzLXZQqC0a8MHJTL5kNTshAKl9kJqOvFYaI=&xsec_source=pc_comment)

做市回测用的啥框架

03-27山东

赞

2

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

hftbacktest

03-27中国香港

赞

回复

展开 1 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/645b7e371fc3de4c930eff9d.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/684c1f83000000001d00911e?channel_type=web_profile_page&xsec_token=ABBdA-PH96I2IZBruNnwb8HcbXfsc3-ma_w-N3-8VjnG8%3D&xsec_source=pc_comment)

[明](https://www.xiaohongshu.com/user/profile/684c1f83000000001d00911e?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABBdA-PH96I2IZBruNnwb8HcbXfsc3-ma_w-N3-8VjnG8=&xsec_source=pc_comment)

所谓的高频，感觉是在给broker 打工

04-01中国香港

赞

1

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

股票是的

04-02山西

赞

回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31uhm0kvk2q605pceef683ijmk6o0j78?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/658e73cc000000002001ca76?channel_type=web_profile_page&xsec_token=ABFNbQkI5vxhQBjDz66afP1lBVfe1UZdyDhH7X3m3hyms%3D&xsec_source=pc_comment)

[uuuup](https://www.xiaohongshu.com/user/profile/658e73cc000000002001ca76?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABFNbQkI5vxhQBjDz66afP1lBVfe1UZdyDhH7X3m3hyms=&xsec_source=pc_comment)

既然说是一行代码起了关键作用，大胆猜一下是对加了某个特殊的处理步骤，再套用某些经典算法![](https://picasso-static.xiaohongshu.com/fe-platform/d1a34cf8aeac526d36890d3e8f727192a6808ecf.png)

03-28广东

赞

1

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

你知道的太多了

# zero copy

![[Pasted image 20260412071432.png]]

# 碎碎念1
![[Pasted image 20260412073745.png]]
![[Pasted image 20260412073813.png]]
![[Pasted image 20260412073823.png]]


从4月份开始入门高频做市策略到现在，这是我个人感觉最最最充实的大半年。昨天也是久违的睡足了6个小时，觉得应该写一篇日志，就当一个阶段性总结。 高频做市血泪史的系列我不打算继续更新了，这一篇就作为小红书部分的收尾。其实转念一想，还是很感谢小红书的、带我认识了很多好的朋友一起交流比如ricky，海东老师，北面，还有云梦量化的朋友们，在我学习量化的路上给了我很多的指导。也认识了很多合作伙伴，机构，做市商同行，甚至为我核心团队添加了新鲜血液！期待未来一起拿到不错的成绩～ 不更新的原因是，我不是一个特别喜欢发社交媒体的人，很多时候发小红书，是想找个地方分享一下现在我做到哪了，有哪些困惑。有时晚上睡不着，翻来覆去想的问题都是哪些。我不喜欢显摆或者炫耀，只是单纯的想分享心路历程。最近到达了自己的一个milestone，觉得应该静下心来好好沉淀一下。 目前一些进度 - 规范化了策略设计流程，调整优化了风控、新开发的策略也上实盘了，同时也发现了高频做市的重要盈利点： 在解决了网速- infra稳定性 - 整体延迟和风控， 输出jitter和regime control后的，最重要的部分，一个高频策略为什么能盈利的核心：定价。 有点后悔到现在才想明白定价模型的重要性，一个强的微观定价模型是有效策略的根基，远比十个alpha有效。 定价标准+网速，说明你能在最快或者比其他人快的时间窗口有更多选择权。 而窗口失效-到策略改价，则是框架，策略执行层，代码的鲁棒性一体化的集中呈现。也就是如果定价能做到100分，在窗口内策略执行层和风控只有及格分也能盈利。 那么如何找到一个强的定价模型应该是高频做市第二阶段的重中之重。其实可以透露一下现有我知道有效的几个思路：1 理解稳定币。稳定币的定价机制是美元，但是也是其他大流动性稳定币定价的平均加权 2 理解leadlag效应 - 流动性低的clob会归因性跟随高的clob 3 理解alpha反转 - 如果所有人都在orderbook竞争，obi加权是alpha 的必然。那么排队顺序上打不过大做市商，因为定价不如人家的速度快，就一定是关注alpha的反转。关注固定时间窗口“fresh price”的确定性。 目前有效策略已经开发了两个了，其中不免有短期预期十分优秀的。但是共享的相关特征都是强定价。于是分享自己的这个发现，期待未来与大家的更多讨论和交流。

2025-12-23

共 33 条评论

[![](https://sns-avatar-qc.xhscdn.com/avatar/411309441983cdc4ce073ff8da73b5b6.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5754bcce346094720e03ce30?channel_type=web_profile_page&xsec_token=ABf9JoKg3z0jF1p0RBEuTPIad0Z8cpUl3YnPvGm8T_6E0%3D&xsec_source=pc_comment)

[人生](https://www.xiaohongshu.com/user/profile/5754bcce346094720e03ce30?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABf9JoKg3z0jF1p0RBEuTPIad0Z8cpUl3YnPvGm8T_6E0=&xsec_source=pc_comment)

自己做的话这得多快的服务器

2025-12-23西班牙

1

3

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

c7i

2025-12-23河南

5

回复

展开 2 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31jj12kiaiu005ousiar9i22gij0c02o?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/63dc92b60000000026010850?channel_type=web_profile_page&xsec_token=AB7KxiCadbkYqF0Z53MVMXA_hZTkLrzmXmRnrXOT8yPY8%3D&xsec_source=pc_comment)

[td](https://www.xiaohongshu.com/user/profile/63dc92b60000000026010850?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=AB7KxiCadbkYqF0Z53MVMXA_hZTkLrzmXmRnrXOT8yPY8=&xsec_source=pc_comment)

想请教下这里的定价模型与alpha的关系怎么理解

2025-12-24广东

赞

4

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

定价是现在fair到底是多少 alpha是预测未来大概会往哪方向走。有时候forcast mid可以是fair

# 碎碎念2

高频量化血泪史 - 停更！但是分享一下小技巧
这一篇是最后的部分了。分享一个和北面老师聊完后的工程学tricks。
这里做的是单交易所双腿（不做跨所对冲）的做市版本。
先说跨所对冲的优缺点：
优势：delta中性，灵活。收益稳定。可以做双边maker的XEMM，也可以做一边maker一边taker的Arbitrage模式。策略容量大，能跑大币对，9-8 figures available
缺点：套利机会越来越机会少，竞争激烈。同 alpha- 价差 用的人越来越多。每次下单量太大，有时不一定单腿总fill，形成头寸暴露。两所仓位保证金平衡起来麻烦，总涉及互相转账的问题。

我直接一个头铁拥抱尾部风险搞单所双腿了。

下面说tricks
1. 框架上，我们团队花了将近一个月，把线性的计算改成共享内存的微服务了。这样绑核了算数据，算完了以后共享内存所有策略都可以调用。
2. 服务器做master - slave node rpc控制，所有子账号绑定elastic ip。用aws eni 开单独接口分行情，一个接colo听 一个eni开多elastic 剩下的用普通ip下。每个eip绑一组api，绑定到多子账户多实例。这里注意，eip不够的话需要和aws单独申请他们的quoting service
3 由于单个ticker/symbol盈亏不可控、面板调用pid，每个小资金100-200 分开跑不同ticker。哪个ticker突然在滚动窗口内突然盈利很多，就集中跑这个ticker，直到cap 打满。

这样做的原因是：思考一下策略有挂单和撤单-当定价决定撤改单，说明这里是个坏价。next price move是个fresh price。如果多个sub打一个ticker，总有instance能挂到 fresh price靠前的队列，平均了网络jitter和队列加入成本。

然后说一个头铁做单所的库存思路。
首先持仓有风险这个大家是公认的。持仓时间越长theata越大
但是但是但是。我们实际是可以控制库存区间的。我的做法是让持仓库存小于每次的order qty，这样的好处就是仓位每次cross 0，正负号变了，我就可以认为这比头寸结算了，进入下一个循环。库存大于一个order qty 就强skew清掉，直到只剩一个小于order qty的小库存。
篇幅有限，写的比较不全面。很多没解释到位的地方大家用ai看一下吧

---


# strategy execution

这一篇讲做市业务层，通常策略执行层在定价之后：

1. 首先是计算reservation price 。这里算当前的公允价格。前面帖子写了，这个板块主要告诉你应该围绕哪个价格去quote。
2. 策略的执行层就是下一步。围绕着定价怎么挂单，挂多少，怎么撤单，什么时候撤单。做持续报价或者狙击择时，有仓位的情况怎么加减仓位，是否跨时间。跨市场，跨交易所执行等。这里我们拿单市场为例，列举我个人测试下来觉得有效的方法： a. AS Model，我认为是跳不过去的一个库存模型管理。position mode调整为oneway，让策略只管理买卖。根据交易强度Kappa 和波动率Volatility计算动态的 spread + 自适应订单 order amount管理。 b. Position Executor- 仓位执行器和 tripple barrier。这也是比较经典的方法。参考Advances in Machine Learning这篇论文，看我的配图2 可以大概了解执行逻辑。我们把所有订单分为两类。入场订单“entry”和出场订单“exit” 。我们给每个entry fill规定三种exit方式。达到TP 止盈；达到SL止损；达到TL，持仓时间过长平仓。当然中间所有的参数都可以做动态拟合和online learning。还可以做魔改，比如SL硬止损设置为移动盈损停止。举例我TP设置5% SL设置-1% 如果fill之后pnl先上涨了3% 又回撤1% 则直接market price离场，锁定2%利润。或把止损订单做成限价的chasing order而不是market c.动态网格， 这个方法在hftbtest的示例脚本中有给到，先说优点：把quote price align到grid可以明显规定固定盈损距离，波动套利，符合做市商逻辑。缺点也很明显：持仓调整过于笨重，会减仓强关联于fill rate，以及如果fillrate太低会被交易所减api权重。如果账号是高rebate结构和vip等级则好用（大于-1bps的rebate，比如有些usdc或新币有额外奖励这种） 其他的执行方法也有很多，单市场的还有dca，glft，gp模型之类的。跨市场的还有XEMM跨所做市、套利等。篇幅有限我们留着以后讲。 ps：管我要论文的小伙伴如果还没收到别着急，人太多了我都是攒齐一批一起发～

# high freq

上回说到负费率对高频做市的必要性。今天来聊聊如何写一个做交易量的策略脚本思路，从而申请到交易所的maker负费率。（使用btcfdusd，但是由于该币taker量的特殊性，扩展到usdt币对则无效） 我们以hummingbot作为框架举例参考写一个script。结构主要分为几个部分： 首先init 的部分就不做赘述了，准备库存：一半大饼现货一半fdusd（需要的话可以用衍生品对冲掉现货价值）。然后开启，等待连接器就绪、获得市场数据、等待数据加载完成… 主要我们讲一下执行层： 1.定价alpha 因子我们使用 OBI 订单簿不平衡，经典的高频因子，市场足够随机的情况下，价格会倾向于往阻力小的方向移动。init结束后，我们先抓取orderbook，计算一个buckect的数据，排队轮换比如100个snapshot总量。 平滑计算后添加到mid price上。这样我们就有了一个adjusted_mid_price。设置一个alpha系数c1用于调整放大比例。之后的挂单参考这个fair value，如图二所示，回测胜率不错。 2.订单简易定时刷新。由于btcfdusd的交易强度，使用事件驱动计算速度会吃不消。我们直接用hummingbot里的scriptstrategybase里的方法get mid price（直接实时抓取bookticker的best bid/ask 和/2 然后rounded向上取整）定时一秒或两秒刷新，对这个计算后的fair value进行下单。 3.设置初始的 spread 比如0.01%或者更窄 经验丰富的也可以直接使用tick_size来实现。注意盘口保护。由于只有maker才是0费率，所以一定要使用time in force = GTX 来确保limit_maker 这样如果post only的挂单因为延迟或者c1系数太大跨盘口就会立即被交易所退还。同时双重保护，设置bid_price = min(bid_price, best_bid)ask_price = max(ask_price, best_ask) 4 skew平衡库存可选，多级挂单可选。止盈hangorder和ttl可选。 欢迎DM，期待大家积极交流评论提供新的思路共同学习。有不对的地方欢迎指出。 编辑于 2025-08-13 共 42 条评论

好好睡觉呢 请问一下 这样刷交易量的cost大概能做到多少呀？ 2025-08-13美国 赞 7

MS 作者 600刀0 cost做了1100w现货一天 2025-08-13加拿大 5 回复 展开 6 条回复

小小小小的野生quant工 小所流动性不好，感觉负费率也很难做唉。挂单很容易被adverse take，在快速趋势中被吃掉 2025-08-23广西 1 12

MS 作者 对，在小所做的话其实定价就很重要了 2025-08-23日本 1 回复 展开 11 条回复

不会思考的笨蛋 fsusd是不是不计入交易量啊，貌似听人说只有有手续费的交易对才算 2025-08-13广东 赞 2

MS 作者 开通做市商账号之后不计算，普通用户一开始刷的时候计算

---
自己开始做高频三个月了，现状是赚赚赔赔、还在给盘口交学费。但是有一些得到认证的思路。怕自己忘了于是写第一篇小红书日记，记录一下高频做市策略的的几个心得。也是从头到尾踩过的坑，Bullet Point一下（只探讨思路，欢迎同行指正）： 1 . 网速大于一切。先说以Binance为例，标准就是需要服务器 aws 东京az1 的 c7i + colocation ws白名单。速度要达到tcp nping 0.2-0.6ms 的 rtt才可以。这个不做过多赘述，延迟会挂到坏价，而且据说colo的orderbook和public不一样，100ms有两个切片（据说），下单最好用ws（不得不说bn这点很良心，其他很多所都需要vip4以上才能用ws post）。 2. maker rebate，这个是必须的。-0.3bps以上才有肉眼可见的利润。 3. 库存惩罚 inventory skew 这个就是需要根据你的持仓数量来判断是否要更快速的清理仓位，衍生品的话多空的目标期待都是0持仓（不加任何因子的话 inventory skew会直接成为逆向选择的最大原因，因为你总是倾向于别人更快take 自己的单子而清理仓位，所以对象价格一定会更近，甚至亏损出货） 4. hedge or oneway？ 这是个非常有趣的话题，很多同行说hedge不赚钱。我一开始的思路是，无论多空、给每个仓位一个固定的止盈，这样只要胜率足够高就行了，但结果事大概率扫止损（虽然我用chasing order的办法追限价单，但还是损spread），而且会扛库存。虽然可以通过类似as模型做库存的delta neutral，但是双向持仓的问题还是保证金占用率和平仓的问题也会导致资金利用率不够。hedge能赚钱吗？能，前提是一定要计划好盈亏比。止盈要大一些。 但是为了拉高资金利用率和足够高的资金换手率，目前还是选择用oneway 5. 价格模型/alpha因子🗣️我是从这里开始盈利的。如果做完全中性的市场就很容易被逆向选择。毕竟做市商是其他所有散户，量化，套利，机构的对手盘。被逆向选择不要太痛苦。但是：如果加了因子的话，就可以合理控制或者减小库存惩罚的副作用，实现“临时扛库存，带着价格拉高盈利出货”的效果。具体的因子和价格模型实现不透露了，有兴趣可以私聊。 先写这么多，欢迎大家纠正错误，给指明方向，交流学习，共同进步。

接上一篇日记，由于发出来反响出乎意料的好，也结识了很多同行朋友一起探讨策略优化的方法 。今天来聊聊高频交易的整体宏观架构。今天写的这一篇想在于对上一篇笔记更多细节的补充。 个人全流程的做下来HFT，一共需要这几个模块，并在每个在细节上做的非常优秀才可以盈利。分别是 基建 - 语言框架 - 策略 - 回测调参/ml/rl - 风控监控 1. Infrastructure 基础架构，包括不限于colo ip服务器，网络调优。服务器线程优化/ ena/ 内存缓存击中率优化/ dpdk协议等。位置上还是以aws为例，可以定期刷新创建新的ec2实例，看看能不能欧一把，排到撮合引擎机房同楼层或者同机房的实例。跨所策略要考虑租用海底光缆或者微波塔专线。 2. Framework 策略执行框架，语言，技术选型（通常使用C或者Rust，原因安全，速度和内存GC回收机制）我们团队自己的框架使用的是Rust，无锁通讯 + ring buffer 提速 3. Strategy Execution 策略执行细节至上！ 包括但不限于选币机制，什么市场/ 择时以及参考的因子。100个细节只做了99个都会导致无法盈利。虽然我觉得“盘感”这种东西在order book的micro structure层面站不住脚，但是有总比没有强。可以辅助优化具体的策略执行。 4. Backtest, ml & rl 回测模块/机器学习/强化学习。量化的本质是数据科学，虽然HFT（这一块很多同行持不同意见）一定是速度优先。所以特征工程和因子挖掘一直是我的短板。一些朋友观点认为微观行情的随机性足够，只要能抢单子就能赚钱。另一些观点认为头部做市商肯定以各种各样的方法抢到盘口一档最佳排队位置。没有军备竞争的量化个体户同台竞技包亏的。不如设计或训练更合理的定价模型、成交概率模型，价格方向预测模型来提高胜率，更优秀的库存管理做市模型来规避风险。我觉得两者都有道理。 5. Risk & Monitor 风控这一块，很多能交易的pair有容量限制，所以很多需要多个子账号操作。个人倾向于自己设计 rpc 监控所有子账号盈损情况。并且设计库存模型/ 规定交易时段以及有毒订单流检测等避免极端行情。毕竟少亏就是多赚嘛。 篇幅有限，后续会对每个板块做更深入的拆解。如有错误欢迎评论区指正和交流。 [#量化](https://www.xiaohongshu.com/search_result?keyword=%25E9%2587%258F%25E5%258C%2596&type=54&source=web_note_detail_r10) [#高频交易](https://www.xiaohongshu.com/search_result?keyword=%25E9%25AB%2598%25E9%25A2%2591%25E4%25BA%25A4%25E6%2598%2593&type=54&source=web_note_detail_r10)

2025-07-31

共 87 条评论

[![](https://sns-avatar-qc.xhscdn.com/avatar/630431713137e2f499f99dbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cce6096000000001701e2d5?channel_type=web_profile_page&xsec_token=ABRzG__RVSHpOeH4f0YNbm0G_zrkJJP2ApilxEsEHvNpw%3D&xsec_source=pc_comment)

[Church](https://www.xiaohongshu.com/user/profile/5cce6096000000001701e2d5?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABRzG__RVSHpOeH4f0YNbm0G_zrkJJP2ApilxEsEHvNpw=&xsec_source=pc_comment)

大佬这163us是哪段延迟

2025-07-31中国香港

1

9

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

策略执行计算，从本地收到ticker到算完下一次下单的ts发出去

2025-07-31加拿大

2

回复

展开 8 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/1040g2jo31ine7nh71a6g5p5m20ak6i2v9edq0ug?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/64b61015000000001003485f?channel_type=web_profile_page&xsec_token=ABf4JvAYap6gFqXFsa7zmJi6G_HhP9bwU8Gp5-f0pj0MQ%3D&xsec_source=pc_comment)

[等一只黑天鹅](https://www.xiaohongshu.com/user/profile/64b61015000000001003485f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABf4JvAYap6gFqXFsa7zmJi6G_HhP9bwU8Gp5-f0pj0MQ=&xsec_source=pc_comment)

同行，其他市场的做市商，请问币圈双边都filled的概率高吗

2025-07-31广东

赞

2

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

成交概率模型、或者每个仓位设置一个ttl生存时间。不fill或者单腿偏离太多就当库存惩罚解决掉

2025-07-31加拿大

4

回复

展开 1 条回复

[![](https://sns-avatar-qc.xhscdn.com/avatar/615864f5581d0a434da55474.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5d499ff0000000001202804b?channel_type=web_profile_page&xsec_token=ABkZbcbwf8ABUGbV3C66_0ALvAI-KDaN6jzX4zOF_XIRo%3D&xsec_source=pc_comment)

[薛定谔的猫](https://www.xiaohongshu.com/user/profile/5d499ff0000000001202804b?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABkZbcbwf8ABUGbV3C66_0ALvAI-KDaN6jzX4zOF_XIRo=&xsec_source=pc_comment)

大佬，高频交易，交易用什么途径的，api开仓吗，代码提交下单指令，怎么让它以最近速度让平台把你单子成交呢？

2025-08-01广东

1

4

[![](https://sns-avatar-qc.xhscdn.com/avatar/63d2ad82904bb17aa5889bbd.jpg?imageView2/2/w/120/format/jpg|imageMogr2/strip)](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_profile_page&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE%3D&xsec_source=pc_comment)

[MS](https://www.xiaohongshu.com/user/profile/5cd2e0f8000000001100968f?channel_type=web_user_page&parent_page_channel_type=web_user_board&xsec_token=ABqVAAoSyTfARklRcwnt87LTl9THzEk60yjTRzCPWNgfE=&xsec_source=pc_comment)作者

API websocket，最快的速度需要多方面优化：1申请平台白名单 对撮合引擎排队有帮助 2网速 3策略框架处理速度 4 服务器
