CAP2CONV_SYSTEM_PROMPT = '''The task is to generate a conversation between two people.
One person is watching at a video, and the other person is asking questions about the video.
What they see will be provided below with some sentences.
When object position information is provided, ask about the position of objects in the scene, including the absolute position and relative position with other objects.
Include at least one complex question that requires reasoning and thinking.
Only include the questions with certain answers that one can answer with the provided sentences.
Make the QA sound like they are seeing the video.  Do not use any words that may sound like looking at text instead of images, like "specify", "mention", "description", "text", "provided information", "sentence", "caption", etc.  Use words like "see", "look", "view", "show", etc.
Format each QA pair in a single line as a JSON dictionary.  Do not include any other explanation.'''

CAP2DES_SYSTEM_PROMPT = '''You are an intelligent assistant that can understand video information through text descriptions. You can understand the overall content of the video from the title of the video, the caption of the video. Please describe the video you saw through the information given above. Don't mention the title in your description. Don't copy the original caption. Do not separately describe which objects are included in the video. It is necessary to integrate object information into your description through adjectives or attributive clauses. This description should be between 150 and 200 words.'''

CAP2REA_SYSTEM_PROMPT = '''\
You are an AI visual assistant that can analyze a single video. You receive a title of this video and a caption of this video, each describing the same video you are observing. 

The task is to use the provided title and caption, create a plausible question about the video, and provide the answer in detail.

Create complex questions beyond describing the scene.To answer such questions, one should require first understanding the visual content, then based on background knowledge or reasoning, either explain why things are happening that way, or provide guides and help to user's request. Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first.

When using the information from the caption, directly explain the scene, and do not mention that the information source is the caption. Always answer as if you are directly looking at the video. 
'''

CAP2REA_FEWSHOT_SAMPLE = [
    ['''\
[Title]
Woman Pranks Sister by Covering Inside of Her Whole House in Aluminium Foil
[Caption]
This woman had gone on a vacation. However, she was shocked when she entered her house on returning. Her sister had covered her whole house with aluminum foil from inside to prank her. She laughed uncontrollably as she saw everything covered in the foil.
    ''',
    '''\
{"question": "Given the sister's initial reaction of uncontrollable laughter upon discovering the prank, how might this prank affect their relationship in the long run, considering psychological and social aspects?",
"answer": " From a psychological perspective, humor plays a significant role in maintaining healthy relationships. The sister's reaction of laughter suggests that she found the prank amusing, which could enhance their bond. Shared laughter can increase feelings of intimacy and social cohesion, indicating that the prank may have strengthened their relationship.

However, factors such as the frequency and intensity of such pranks, the personalities of the sisters, and their previous relationship dynamics can play a role in the long-term impact of the prank. If pranks of this scale become a regular occurrence, or if they begin to cross personal boundaries, they could lead to feelings of annoyance or stress, potentially straining the relationship.

From a social perspective, the prank might become a shared story between the sisters and their social circles, further reinforcing their bond. It could serve as a humorous anecdote that the sisters bring up at gatherings, enhancing their social interactions.

Nonetheless, the long-term effects of such pranks largely depend on how both parties perceive the incident. If they both view it as a light-hearted jest, it could positively influence their relationship. However, if the prank becomes a source of resentment or discomfort, it could potentially damage their relationship. As such, communication is key in ensuring that such pranks contribute to a fun and healthy relationship rather than detracting from it."}
    ''']
]

CAP2DES_FEWSHOT_SAMPLE = [
    ['''[title]\nGuy Scratches Head After Landing Perfect Bowling Strike\n[Caption]\nThis guy scratched his head in confusion after making a mind-blowing attempt at bowling. He swung his hand to release the ball but accidentally tossed it towards the gutter. However, it spun and turned at the side edges of the lane and then struck all pins in one go.\n[Object Infomation In Time Order]\n00:00-00:02 : a black ball:[3, 327, 500, 816];a hand holding a black frisbee:[0, 605, 516, 1076];a large window to the outside:[459, 39, 606, 200];white cabinets on the wall:[2, 15, 472, 264]\n00:02-00:04 : the ball is round:[3, 504, 426, 1036];white doors on cabinets:[0, 0, 489, 231];floor is made of wood:[3, 210, 607, 1069]\n00:04-00:06 : black plastic frisbee in mans hand:[0, 746, 140, 910];red shirt on man:[1, 278, 202, 640];black speaker above the desk:[479, 20, 599, 152];white door with black doorknob:[474, 190, 524, 339]\n00:06-00:08 : man is bending over:[1, 366, 335, 848];a black shoe on the man:[85, 790, 168, 847];a man is wearing a red shirt:[68, 398, 304, 598];black ball on the floor:[349, 475, 386, 513];the man is wearing shorts:[3, 503, 111, 704]\n00:08-00:10 : a man wearing a red shirt and khaki shorts:[0, 231, 317, 913];a black shoe on a man:[1, 818, 106, 912];the floor is made of wood:[1, 418, 606, 1073];a white door with a black doorknob:[448, 300, 500, 448]\n00:10-00:12 : man wearing a maroon shirt:[118, 182, 431, 1073];black television hanging from the ceiling:[244, 93, 402, 212];a black shoe on a man:[152, 982, 274, 1070];red letters on shirt:[228, 446, 298, 508]\n''',
    '''In the video, we see a man wearing a maroon shirt and shorts standing in a bowling alley, holding a bowling ball. First, he swings his hand to release the ball but accidentally tosses it towards the gutter. Next, the ball spins and turns at the side edges of the lane, seemingly heading towards the gutter, but suddenly changes direction and heads towards the pins.\nThe ball strikes all the pins in one go, resulting in a perfect strike. The man looks confused as he scratches his head. The man is surprised and shocked by what just happened, and he raises his hands in disbelief.\nThroughout the video, we can see a red tee-shirt with red letters on the man's shirt. We also notice that the man is wearing black shoes . Additionally, we can see a black television hanging from the ceiling in the background.\nIn summary, the video captures a surprising and unexpected turn of events in a bowling alley. The man's accidental toss ends up with a perfect strike, leaving both him and the viewers amazed.'''], 
]