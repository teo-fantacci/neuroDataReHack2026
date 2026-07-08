# Notes and project
- [[IBL_Brain_Wide_Map_dataset_summary]] 
- [[NeuroDataReHack_project_notes]] 
- [[Janelia - first project plan]] 

### Claude prompt
ask to do a thorough summarize of the following papers:
1) 2019 harris connectivity 
2) 2023 Engel geometry 
3) rSLDS  

# Dataset 
- 00049 IBL Brain Wide Map 
	- https://dandiarchive.org/dandiset/000409
	- small description -> note 
# Scientific questions
- given the vast set of area (though most of them are involved in the decision process as shown in the paper of reference Nature 2025), can we infer the flow/sequence of attractor states that lead to the decision starting from the visual input? 
	- Not exactly but we can use tools to infer the most probable observed from the spike activities: 
		- HMM 
		- rSLDS
		- interregional interaction techniques
	- use neuralflow to show that the dynamics is an attractor like shared and is common across areas. The tuning curve geometry instead is area specific 
- appraoch in 2 ways 
	1) temporal sequence of attractor states for information processing 
		1) HMM, rSLDS, 
		2) compare with beahviours (reaction time, choice vs error)
	2) different geometries in different areas jointly participate to the final goal but with different type of computation (aka neural geometry)
		1) T-PHATE, CEBRA, 
		2) take the reference of Gallego lab (Fortunato et al. ) and the Tatiana 2023
1) framed as _distributed temporal sequences within populations_, not as a clean anatomical pipeline
2) the same latent attractor dynamics of the decision are realized on geometrically distinct, area-specific manifolds
# Temp Claude
Now we can start with notes in an .md file were you can summarize the main ideas. At the end we will merge those idea in the project plan. for each part summarize the explanation you gave me (not too much, keep well the concept) and at the end a bullet point with pros and cons. 
1) put the MSR analysis you have done (rememebr that we have to double check if the units selected are also stimulus/choice selective, add the list of the options you listed)
2) note the binarization and the limit of the bin size. note that we will use only good units 
3) about the ELA. I understand the point you make, the normal application is on the whole recording. But what if we partition them in the time scale we know are necessary for the information to flow form one area to another? For example the V1 takes a certain amount of time to pass it to the V2, then V3 etc. If we build the hierarchy as subset of time interval that move the state from one pattern of configuration to another? or even create a ELA for the whole recording and then build it also for the small time window and see how the state trajectory evolve to reach the final/global minima that correspond to the same pattern of activity for that specific area in that specific time window + the activity of the rest of the population. 
	1) does it make sense wrt the small basin of attraction to the large one? maybe not, is just speculation. Tell me what you think about. 
	2) maybe use the HMM and/or rSLDS as the input to make the state evolve across the different configuration
	3) but I understand that for the flow of information is better the HMM or rSLDS for the matter of time. I agree on using ELA as the structural view. Keep it as a note for follow-up.
4) I like this view "within simultaneously recorded populations, decision-making unfolds as a temporal sequence of metastable states, and the cross-area latency structure of those states reflects distributed information flow" with the caveat of distributed and parallel processes 
5) about the anatomy and the flow: 
	1) you said "synaptic/cell-type-specific directed wiring and the _functional_ direction of flow are much less settled". This is probably what I would like to tackle in my project (mostly the functional part given that the directed wiring needs different experiments). Do you think rSLDS can be used for that? Maybe including priors of the mesoscale anatomy. 
	2) 
	3) 
# Techniques 
- Neuralflow (M. Genkin)
- Cell assemblies (E. Russo)
- CCA across regions 
- MIC (M. Celotto)
## possible approaches
### Attractors and sequence of attactors
1) multi-scale / tree-structured switching dynamics (tree-structured rSLDS) 
2) heteroclinic sequences / "winnerless competition" 
	1) trajectories that chain through a series of saddle states, associated with Rabinovich and Laurent's work on olfactory sequences (I'm recalling this rather than having pulled a specific paper in this session, so treat it as a lead to verify, not a settled citation 
3) nested energy landscapes where coarse attractors contain finer sub-attractors (a Hopfield-style idea) 
4) 