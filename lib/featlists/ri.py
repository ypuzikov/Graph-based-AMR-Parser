#!/usr/bin/env python
# -*- coding: utf-8 -*-

feats = [

"bias",

"hc", "hc1q", "hcd", "hch", "hc1c", "hc2c", "hcll",
"tc", "tc1q", "tcd", "tch", "tc1c", "tc2c", "tcll",

"hw", "tw",
"hp", "tp",
"hp_tp",

"ppath", "dpath",
"path_len",
"dist",

"h_headw", "t_headw",

"h_headp", "t_headp",
"h_headp+t_headp",

"h_headd", "t_headd",
"h_headd+t_headd",

# experimental

# "hn", "tn",
# "hn+tn",

# "t_head_in_h_head_args",
# "h_head_srl_role",


# "h_head_ispred",
# "t_head_ispred",
# "hw_preds",
# "hw_args",

# "hw_is_pred_of_tw",
# "hw_srl_role",

# "hppath",
# "hdpath",
# "hppath+hdpath",
# "tppath+tdpath",
# "hppath+tppath",
# "hdpath+tdpath",
# "hppath+hdpath+tppath+tdpath",


# "tppath",
# "tdpath",

# experimental:

# "hsrl",
# "tsrl",
# "bonc1",
# "bonc2",
# "bonc3",
# "bonc_all",



# Result:

#hp1_c
#hp1_c1q
#hp1_cd
#hp1_ch
#hp1_c2c
#hp1_cll

#tp1_c
#tp1_c1q
#tp1_cd
#tp1_ch
#tp1_c2c
#tp1_cll

# Results: train1000, dev_full: 44.43, test_full:. Nevertheless, it is a wrong way!

#hp1_c1c -
#tp1_c1c -
#hdpath -
#tsbl_cs - very bad!
#hppath_dist -
#tppath_dist -
#tppath
#tdpath

#hl_tp -
#tl_tn -
#tsbl_labs -
#hp1_ch_num
#hp1_c1c -
#tp1_c1c -
#hp1_cd +
#hp1_ch +
#hc3c
#tc3c
#hp1_c1c
#hp1_c3c
#tp1_c3c
#hp1_cs
#hp1ppath
#hp1dpath
#hn1w
#hn1p

#hp2_c
#hp2_c1q
#hp2_cd
#hp2_ch
#hp2_c1c
#hp2_c2c
#hp2_cll

#tp2_c
#tp2_c1q
#tp2_cd
#tp2_ch
#tp2_c1c
#tp2_c2c
#tp2_cll

#hp1lab
#hc+hp1c+tc
#hp1_c+hc+tc

#tp2_c
#hp1_c+hc+tc
#hctypestr+hp1ctypestr
#tsbl1_c
#hp1dist
#hp1ppath
#hp1dpath
#hp1lab
#hp1_c+hc+hp1lab

#hp1w
#hp1p
#hp1ctype
#hp1p_tp1p
#hp1w_tp1p
#hctype -
#tctype -

#num_hpid
#num_tpid
#hgpid_c
#tgpid_c
#hpid_c+hpid2_c
#tpid_c+tpid2_c
#hpid2_c
#tpid2_c
#hidsbl_c
#tidsbl_c
#hheadw
#theadw
#hheadp
#theadp

#hc_verb -
#path_length -
#hc_tc_pmi -
#dpath_hctype_tctype
#ppath+hc+tc
#ppath+dpath
#ppath+hp+tp
#dpath+hw+tw
#hw_tw -
#tw_tp -
#hp_tw -
#hc_tc -
#hw_tp
#hw_hp
#hp_tw
]