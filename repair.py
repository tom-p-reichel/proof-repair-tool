from transformers import AutoModelForCausalLM,AutoTokenizer,StoppingCriteria,StoppingCriteriaList,BitsAndBytesConfig
from typing import NamedTuple
from coqtop import CoqProcess
import more_itertools
import asyncio as aio
from pathlib import Path
import numpy as np
import math
from tqdm import tqdm
import pickle
import torch
from dataclasses import dataclass
import prism.util.alignment as  align
import random
import re
from functools import lru_cache


prefix_alignment  = align.align_factory(lru_cache(maxsize=30000)(lambda x,y : align.fast_edit_distance(x,y)),lambda x: len(x),select_best=lambda D: (D[:,-1].argmin(),D.shape[1]-1),numba=False)


"""
model_base = "./base_model/"

# TODO: this quantization config is not quite right!

"""



import asyncio as aio


class StackManager():
    def __init__(self,prefix,flags,n=1):
        self.stacks = {}
        self.n = n
        self.initialized = False
        self.prefix = prefix

        self.offset = len(prefix)+1

        self.flags = flags

        self.sema = aio.Semaphore(n)
        self.biglock = aio.Lock()

    async def __postinit__(self):
        self.initialized = True
        self.ctxs = []
        for i in range(self.n):
            self.ctxs.append([(),CoqProcess(*self.flags.split(),verbose=False),aio.Lock()])
            stdout,stderr = await self.ctxs[-1][1].run("\n".join(self.prefix), return_stderr=True)

            if "Error:" in stderr:
                raise ValueError(stderr)


    async def evaluate(self,stack):
        if type(stack)==list:
            stack = tuple(stack)
        if stack in self.stacks:
            return self.stacks[stack]

        if not self.initialized:
            async with self.biglock:
                if not self.initialized:
                    await self.__postinit__()

        async with self.sema:
            ctx = next((x for x in self.ctxs if not x[2].locked()))
            ctxstack,coq,ctxlock = ctx
            async with ctxlock:
                ctx[0] = stack
                stdout,stderr = await coq.run(f"BackTo {self.offset}.\n"+"\n".join(f"timeout 1 {x}" if x[0].islower() else x  for x in stack),return_stderr=True)
                output = await coq.run("Show.")

        if "Error" in stderr:
            self.stacks[stack]=None
            #print(f"error encountered running {stack}:\n{stderr}")
            return None
        else:
            self.stacks[stack] = output
            return output



class CoqStoppingCriteria(StoppingCriteria):
    def __init__(self,tokenizer):
        self.banned = []
        self.banned.append(tokenizer("\n```",add_special_tokens=False,return_tensors="pt")["input_ids"][0])
        # this better not be true or else the tokenizer can eat the \n.
        assert(all(".\n" not in x for x in tokenizer.get_vocab()))

    def __call__(self,ids,scores,**args):
        return all(".\n```" in x for x in (tokenizer.decode(x).split("[SEP]")[-1] for x in ids))


class ModelManager():
    @dataclass
    class Request():
        prompt : str 

    def __init__(self,model,tokenizer,environment,batch_size=12,cache_path=Path("~/.cache/yapsrt/vecs.pk").expanduser()):
        self.m = model
        self.tokenizer = tokenizer
        self.queue = aio.PriorityQueue()
        self.env = environment
        self.serving = False
        self.counter = 0
        self.bad_words = [[y] for x,y in tokenizer.get_vocab().items() if ";" in x or "(*" in x]
        self.stoppers = [CoqStoppingCriteria(tokenizer)]

        """
        if Path(cache_path).exists():
            with open(cache_path,"rb") as f:
                try:
                    cache = pickle.load(f)
                except:
                    cache = {}
        else:
            Path(cache_path).parent.mkdir(exist_ok=True)
            cache = {}
        self.m.set_adapter("search")
        new_thms = [(k,v) for k,v in self.env.items() if (k,v) not in cache]

        for k,v in tqdm([(k,v) for k,v in self.env.items() if (k,v) not in cache],desc="vectorizing env"):
            with torch.no_grad():
                cache[k,v] = torch.nn.functional.normalize(self.m.model(**self.tokenizer(f"{k} : {v}", return_tensors="pt")).last_hidden_state[:,-1]).cpu()

        with open(cache_path,"wb") as f:
            pickle.dump(cache,f)

        # pick an order
        self.vec_thms = [k for k in self.env.keys()]

        self.vecs = torch.vstack([cache[k,v] for k,v in self.env.items()]).cuda().half()
        """

    def eval_request(self,request,n=10):

        with torch.no_grad():
            try:
                #self.m.set_adapter("tactic")
                results = self.m.generate(
                    **self.tokenizer(request,return_tensors="pt"),
                    num_beams=n,
                    do_sample=False,
                    stopping_criteria=self.stoppers,
                    use_cache=True,
                    max_new_tokens=40,
                    num_return_sequences=n,
                    bad_words_ids=self.bad_words,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    length_penalty=0.0
                )
            except torch.cuda.OutOfMemoryError:
                return None
       
        tactics = [tokenizer.decode(x).split("[SEP]")[-1].split("\n")[0].strip() for x in results["sequences"]]
        states = [[y.to("cpu").clone() for y in x] for x in results["hidden_states"]]
        seq_scores = results["sequences_scores"]

        del results

        #self.m.set_adapter("search")

        #goal_hash = torch.vstack(pattern_vecs).mean(axis=0).float().cpu().numpy()
             

        return {"tactics":tactics,"hidden_states":states,"scores":seq_scores} #, "hash": goal_hash}


    async def serve(self):
        buf = []
        while True:
            priority,(request,future) = await self.queue.get()
            print("handling priority level", priority)
            # check if it's cancelled
            out = await aio.to_thread(self.eval_request,request)
            future.set_result(out)

    async def evaluate(self,prompt,priority=0.0):
        loop = aio.get_event_loop()
        if not self.serving:
            self.serving = True
            aio.ensure_future(self.serve())
        future = loop.create_future()
        self.counter += 1
        await self.queue.put(((priority,self.counter),(prompt,future)))
        return (await future)


def trim_kvs(kvs,l):
    if type(kvs)==tuple:
        return tuple(trim_kvs(x,l) for x in kvs)
    if type(kvs) == torch.Tensor:
        return kvs[:,:,:l,:]
    return kvs
    

@torch.no_grad()
def sample(model,tokenizer,prompt,temperature=0.60,maxlength=1024):
    logits = {}

    prompt_tokens = tokenizer([prompt], return_tensors="pt")
    prompt_length = len(prompt_tokens.input_ids[0])
    tmp = model(**prompt_tokens, use_cache=True)
    cache = ((), tmp.past_key_values)
    logits[()] = torch.nn.functional.softmax(tmp.logits[0,-1,:]/temperature)
    
    stack = []
    
    while True:
        if tuple(stack) not in logits:
            prefix_length = 0
            for x,y in zip(cache[0],stack):
                if x == y:
                    prefix_length += 1
                else:
                    break

            if prefix_length == len(stack):
                print("really weird at", tokenizer.decode(stack), stack)
                prefix_length -= 1


            tmp = model(torch.tensor([stack[prefix_length:]]),use_cache=True,past_key_values=trim_kvs(cache[1], prefix_length+prompt_length))
            
            logits[tuple(stack)] = torch.nn.functional.softmax(tmp.logits[0,-1,:]/temperature)
            
            cache = (tuple(stack), tmp.past_key_values)

        p = logits[tuple(stack)] 

        stack.append(torch.multinomial(p,1)[0].item())

        if len(stack)>maxlength or (tokenizer.decode(stack).endswith("\n```") and len(stack)>0):

            if len(stack) <= maxlength:
                yield tokenizer.decode(stack)[:-4]
            # trim off the ".\n```"
            stack = stack[:-3]
            prob = 1.0
            for i in range(len(stack)):
                prob *= logits[tuple(stack[:-i-1])][stack[-i-1]]
                logits[tuple(stack[:-i-1])][stack[-i-1]] -= prob
            stack = []

    return "i give up."


def tokenize_glb_chunks(tok,chunks,m):
    """
    return `i` such that tokenizing `chunks[:i]` will not exceed `m` tokens
    """
    acc = 0
    for i,c in enumerate(chunks):
        acc += len(tok(c).input_ids)
        if acc >= m:
            return chunks[:i]
    return chunks


def mkprompt(tok,diff,proof_history,proof_pane, budget=2048):
    # we really want to include this

    if len(proof_pane)>5000:
        return None
    
    budget -= len(tok(proof_pane).input_ids)
    if budget <= 400 :
        return None

    if diff is None: # synth example
        history_budget = budget
    else:
        history_budget = budget/3

    proof_history = "\n".join(tokenize_glb_chunks(tok,proof_history[::-1],history_budget)[::-1])

    budget -= len(tok(proof_history).input_ids)

    if diff is not None:
        # spend the rest on the diff
        diff = "\n".join(tokenize_glb_chunks(tok,diff.split("\n"),budget))
   
    blocks = [
            f"Ongoing Proof History:\n```\n{proof_history}\n```",
            f"Proof State:\n```\n{proof_pane}\n```"
    ]

    if diff is not None:
        blocks = [f"Commit Diff:\n```\n{diff}\n```"] + blocks

    #random.shuffle(blocks)

    return "\n\n".join(blocks) + "\nNext Tactic:\n```\n"


bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.float16
)

base_dir = "/home/tpr/s2loop/"

m = AutoModelForCausalLM.from_pretrained(f"{base_dir}/base_model/",device_map="auto", use_cache=False, quantization_config=bnb_config,
        attn_implementation="flash_attention_2")

m.load_adapter(f"{base_dir}/model/","tactic")

tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/base_model/")


async def repair_proof(sentences,proof_start,proof_end,diff,flags,gpu_lock):
    coq = CoqProcess(*flags.split())
    await coq.run("\n".join(x.text for x in sentences[:proof_start]))
    env = await coq.environment()
    coq.close()

    stack_manager = StackManager([x.text for x in sentences[:proof_start]], flags)
    model_manager = ModelManager(m,tokenizer,env)

    # assuming we won't more-than-double proof length
    stack = []

    state = await stack_manager.evaluate([])

    for _ in range((proof_end-proof_start)*2):

        test = await stack_manager.evaluate(stack + ["Qed."]) 
        if test is not None:
            print("we did it!")
            print(test)
            print(stack)
            break
        else:
            print("not done yet.")


        print(stack)
        alignment = prefix_alignment([x.text for x in sentences[proof_start:proof_end]], stack)

        old_cnt = 0
        proof_history = []
        for j,(x,y) in enumerate(alignment):
            if x is None:
                proof_history.append(f"+  {y}")
                continue
            old_cnt += 1 # used a stmt from old proof
            if y is None:
                proof_history.append(f"-  {x}")
                continue
            if (x==y):
                proof_history.append(f"   {x}")
            else:
                proof_history.append(f"-  {x}")
                proof_history.append(f"+  {y}")
        
        recommendation = sentences[old_cnt+proof_start].text
        for s in sentences[old_cnt+proof_start:old_cnt+proof_start+3]:
            proof_history.append(f"?  {s.text}")

        if (await stack_manager.evaluate(stack+[recommendation])) is not None: 
            # ok, it looks like the original proof still works here.
            # we'll just take that line
            stack.append(sentences[old_cnt+proof_start].text)
            continue

        # we might need a bullet and i didn't train the model to use them.
        # just try them all really fast.
        #for bullet in ["-","+","*","}"]:
        #:    if (await 

        
        proof_pane = await stack_manager.evaluate(stack)

        print(proof_pane)

        if proof_pane is None:
            raise ValueError("previous state invalid.")
        
        prompt = mkprompt(tokenizer,diff,proof_history,proof_pane)

        s = sample(m,tokenizer,prompt)

        print(prompt)

        while True:
            async with gpu_lock:
                tactic = await aio.to_thread(next,s)
            print(tactic)
            # TODO: real search logic...
            tactic = re.sub(r"<LOOKUP>([^\s]+) : .+?<\/LOOKUP>", r" \1 ", tactic)

            if tactic[0].isupper():
                continue

            if tactic[0].islower():
                attempt = await stack_manager.evaluate(stack + [f"progress {tactic}"])
            else:
                attempt = await stack_manager.evaluate(stack + [tactic])

            if attempt is not None:
                stack.append(tactic)
                break
    else:
        return None

    return stack



import time
async def main():
    """
    print(await stack_manager.evaluate(("intros.", "try auto.")))
    print(await stack_manager.evaluate(("intros.","pose 1 as e.", "try auto.")))
    print(await stack_manager.evaluate(("intros.", "pose 1 as e.")))
    print(await stack_manager.evaluate(("intros.", "pose 1 as e.","yabba dabba doo.")))
    """

    start = time.time()
    print(await solve_proof(text))
    print("elapsed",time.time()-start)
    #print("model invoks:",model_manager.counter)



if __name__=="__main__":
    pass
    aio.run(main())
