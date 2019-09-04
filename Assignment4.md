# Assignment 4

##### DD2424: Deep Learning in Data Science

##### Beatrice Ionascu (930705-2143, bionascu@kth.se)



## i. Gradient check

The analytic gradients are checked against numerical gradients computed using the centered difference formula and a custom implemented function that computes the gradients numerically. The relative error is higher than expected, however after clipping the gradients, the RNNs seem to train correctly. One explanation could be that there is an error in the gradient check logic or the numerical gradient computation .



## ii. Smooth loss function

Smooth loss function for a training run of 10 epochs.

![plot1](/Users/beiona/workspace/mlenv/char-vanilla-rnn/plot1.png)

## iii. Evolution of synthesized text

The evolution of the text synthesized by a RNN with learning rate 0.1 and sequence length 25 during training before the first and before every 10,000th update steps when trained for 100,000 update steps.

```
iter: 0 
smooth_loss: 110.471017

dAz1!'H0eUF?�-f9x(s01':j�9�o!�	Z
S?aQOKkI1L4I,Eo6SQHu la;v"UcwwAPdD4WJte
P34C1T�U,y0U_�_mAYortN.;aSquRLER T^2Tktq-4f'B2hmFC	Lv-;ot!AZh7gJ}sRw}K;-9u�MJa1SMi}fm.,shUNWDAZd9Dt��.�3s"4TKRR�XFiOMf�c1yFrCz"


iter: 10000
smooth_loss: 55.096976

"I'k.  "I rtoos co er the aswistidn.  and.  Ncet.
"Augttilt sid Bu think aptank. "narl insath yeed a tond.  She at hit belpsangnd, er whelly.  "Dowene tito the tries cougy sary natpurafe, comdy my ple


iter: 20000
smooth_loss: 51.558779

es, Seimick- S a way Firof Harrid then id jash ky the une pitwen.  Ma couker liery. said, be bley tued alinked Harry bot teared, thel meruke, aof do Steakf, ithous detangusk tollie fes courlofe cheefa


iter: 30000
smooth_loss: 50.276836

urs cerd, ak
reas ve mise in tht ghere pack) Siswaspoutor, midblis, hat ase were eving onetwentes in to ens way alionerdere wiy andis trimchered quen to aible whis were tlew thamistoeds, what loby her


iter: 40000
smooth_loss: 49.118762

 the dill gikest wiztirs this not pornedrould et belansed, prort, It worked of fimned undursane fere le mire warwaring to has had - stomenaf, morlaspego; theire ...
Und thime, he sis by oving Vormort


iter: 50000
smooth_loss: 49.740781

! ."
"Mrgameesle wiy oitim, tond bling tevar teacls', abound gontsch tond dise was unticling the foo be quurss mooved they prive heve Harry now toulf curt us he Onorgh, they Ron.
"Therreeching mout an


iter: 60000
smooth_loss: 48.614205

as be was to the hid Harry, bred
"It wholly,"  Ahy Dext Ron eachousched exbling wothen slonand head sangs ladked and tom them erack.  Haurd. Geaving," weed eveny. Tronn?" As juth.
"Lofnaud!"
Amern's t


iter: 70000
smooth_loss: 48.261878

of you, Buching frou oth on to a net a crotles noam was got stume of nop frowfing, you sad I vementor a wanne Harry frre pack Houdang u thoping cermon moge compe intimptar Harry and so and don't fas m


iter: 80000
smooth_loss: 46.707894

rked . . Crom . . . . ... sued, werehursienty. CImmef the was wave to entoro.  Trmaid and f.  "Leat taseden , I gaid Harry-adt frowt he sruntrroor say, averew, He ergecher he congs decar solstercered


iter: 90000
smooth_loss: 47.232027

ng had in no hidgont if!  Harry Worked Waut Vold wasH.  Harmy on As the was nors his of there bexped, the was im of the behth is had hill Cssing.  Shing ortansth steursws hatrrakting abick a peart eni


iter: 100000
smooth_loss: 48.225792

t dow eptrobentore way wayed the booky.  Told mook ap'ordey deredess intilly, whouggrime dan tool, Harry, lopmey agrioss.  Bage at a Gorter and  wer, the moull, the shis the dack to and seen mormiokcr
```



## iv. Passage synthesized by the best model

Passage  from  best  model  with  `learning rate =  0.1`  and  `input sequence  length  = 25`  (`num steps = 433140`, `smooth loss = 42.744658`):

```
hacked and tiniply; a spirting he your in the eyien, the galdling around you they was to her match ingriwn this drastonf Chariaz Wort.
"Oh sood tig gis don't.
"You'm all up righ were me, be, but up, suck the mes, you like could squed sidering the topleeday bell; the seous do "ittle stilling of him poit Bull to thisasey good sought boyingariok a muel out to C'toward a plare.  Onoy, that rove fresing.  Mouh hithir ast to the You look rround feether they sto chell, Ron.  "Havemounto."
"Hoor, my tious.  It, Dumbledn't priches.  Holded the gother. Maviineshor.  The Borcusst low, likewed. . . boding at Mousing at his panse eye.  "Vold her clow this around tures rust stidsing it to Gorniviem was dobored that onkidd up thoufe was a corlly a gloom.  Rougin of been?"  stalkelowsthy," bone the hastfow righed "his go the taper the imblese will wizards," saike!" sagring to Harry unoods the edoucr clook wike or she moods.  "Mngull anxis hand the sdy has myan old ficking to chose oftioned her him som
```

Passage  from  best  model  with  `learning rate =  0.1`  and  `input sequence  length  = 20`  (`num steps = 541624`, `smooth loss = 37.218700`):

```
 yaul sulden at the very lell tomusie you trotchs ald eye and Hermides "The . . .. mut at With they ene, soud, it feromy's ilet go Juskeys erthing tirkly said Ring rely woud .  Ae lut.

no fileed the le?"
Harry Harry tround, I'n ot The drinse.  "Harry Sice rm they tald tot Drane the rown't Magoush's ing the conts Pours, that tarking Harrad, Go Pyovenenting the the'm with intto sparbyeatt.
"Burlyivt.  The.
"WJubter of formione one egce yoused.  Harry fir swere with to you ichid.
"He a more fround he ittath doandly.
"That not bake Drought eoklyling the from ay of tibly beuldessidy arr hen of dic; so dive srick so raund.
"It'ryed hehkor ithar yniffeass me cotect into for a beaker at thouth a shy aft his chints proily tat-thene gire's gyoneres, thay sust rid So for stapide waylly ach-fin.  "I dlecGo that't tastee she sime the seer hagn by julvid the Proord to berim at on doce bus in as hale and this the orn't deen thint ditn-"
ro he his say prapped to steno mug, ut methoulr hick.  "You int
```



Also tested the following combinations which have resulted in a higher smooth loss /  worse results: `input sequence length = 25` and `learning rate = 0.01, 0.09, 0.11, 0.2`.
