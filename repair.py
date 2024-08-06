"""
TODO
"""
# Standard/General Purpose Imports
from typing import Any, DefaultDict, Iterable, cast, List, Tuple, Dict, Optional
from contextlib import contextmanager
from pathlib import Path
import asyncio as aio
import pickle
import re
from functools import lru_cache
from collections import defaultdict

# Tensors and Transformers Imports
import torch
from torch import Tensor, _dynamo
import goodinference
from transformers import AutoModelForCausalLM,AutoTokenizer,\
    BitsAndBytesConfig,PreTrainedTokenizer, PreTrainedTokenizerFast

# Coq Specific Imports
from coqtop import CoqProcess
from prism.language.heuristic.parser import CoqSentence
import prism.util.alignment as  align
from prism.util.alignment import Alignment

# Locally within Proof Repair Imports
from general import ProbsKey, ProbsValue, one_random_choice,\
    process_logits, simplify_whitespace, unseen_test
from stack_manager import StackManager, StackManagerEvaluatesItem,\
    StacksKeys, StacksValues

_dynamo.config.verbose=True

#pylint:disable=unnecessary-lambda
prefix_alignment  = align.align_factory(
    calign=lru_cache(maxsize=30000)(lambda x,y : align.fast_edit_distance(x,y)),
    cskip=lambda x: len(x),
    select_best=lambda D: (D[:,-1].argmin(),D.shape[1]-1),
    numba=False)

ROUGH_SENTENCE_PATTERN = re.compile(r".(?:\s|$)")

type TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast

def trim_kvs(kvs,l) -> Any:
    """
    TODO
    """
    if isinstance(kvs,tuple):
        return tuple(trim_kvs(x,l) for x in kvs)
    if isinstance(kvs,torch.Tensor):
        return kvs[:,:,:l,:]
    return kvs

@contextmanager
def get_search_model(model):
    """
    TODO
    """
    model.set_adapter("search")
    try:
        yield model.model
    finally:
        model.set_adapter("tactic")

type EmbedCache1Items = Any
type EmbedCache1 = Iterable[EmbedCache1Items]
type EmbedCache2 = Tensor | None

if Path("/tmp/vector_cache.torch").exists():
    #pylint:disable=unspecified-encoding
    with open("/tmp/vector_cache_index.pk","rb") as f:
        embed_cache = (
            cast(EmbedCache1,pickle.load(f)),
            cast(EmbedCache2,torch.load("/tmp/vector_cache.torch"))
            )
else:
    embed_cache = (cast(List[EmbedCache1Items],[]),None)

embed_cache = cast(Tuple[EmbedCache1,EmbedCache2],embed_cache)

type FetchEmbeds1 = EmbedCache1
type FetchEmbeds2 = EmbedCache2

def fetch_embeds(model,tok : TokenizerType,
                 thms : Iterable[EmbedCache1Items]) -> \
                    Tuple[FetchEmbeds1,FetchEmbeds2]:
    """
    TODO
    """
    # 1. check memory cache
    #pylint:disable=global-statement
    global embed_cache
    embed_cache = cast(Tuple[EmbedCache1,EmbedCache2],embed_cache)

    index,embeds = embed_cache

    unembedded = set(thms) - set(index)

    if len(unembedded)==0:
        return embed_cache

    # 2. embed them ourselves

    unembedded = list(unembedded)

    vecs : Tensor = goodinference.embed(
        model,tok,
        [f"Theorem {x[0].split('.')[-1]} : {simplify_whitespace(x[1])}." for x in unembedded],
        progress=True)

    vecs = vecs.cuda().half()

    if embed_cache[1] is None:
        embed_cache = (unembedded, vecs)
    else:
        embed_cache = (cast(EmbedCache1,index+unembedded), torch.vstack([embeds,vecs]))

    torch.save(embed_cache[1],"/tmp/vector_cache.torch")
    with open("/tmp/vector_cache_index.pk","wb") as f_for_dump:
        pickle.dump(embed_cache[0],f_for_dump)

    current_thms = set(thms)

    keep = [i for i,x in enumerate(embed_cache[0]) if x in current_thms]


    return [embed_cache[0][x] for x in keep], embed_cache[1][keep]

#pylint:disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
@torch.no_grad()
def sample(model,
           my_tokenizer : TokenizerType,
           prompt,
           env : Dict,
           temperature=0.60,
           maxlength=256,
           p = 0.1):
    """
    TODO
    """

    # we're gonna add fake entries
    env = env.copy()

    sample_thresh = p

    index = list(env.items())

    with get_search_model(model) as search_model:
        index, vecs = fetch_embeds(
            search_model,
            my_tokenizer,
            index)
        #pylint:disable=pointless-string-statement
        """
        goodinference.embed(model.model,
        tokenizer,
        [f"Theorem {x[0].split('.')[-1]} : {simplify_whitespace(x[1])}." for x in index],
        progress=True)
        """

    vecs = torch.nn.functional.normalize(vecs)

    logits : Dict[Tuple,DefaultDict[ProbsKey,ProbsValue]] = {}

    #pylint:disable=invalid-name
    LOOKUP_TOKEN = my_tokenizer.convert_tokens_to_ids("<LOOKUP>")
    #pylint:disable=invalid-name
    UNLOOKUP_TOKEN = my_tokenizer.convert_tokens_to_ids("</LOOKUP>")
    #pylint:disable=invalid-name
    COLON_TOKEN = my_tokenizer.convert_tokens_to_ids('▁:')
    #pylint:disable=invalid-name
    COMMENT_TOKENS = [my_tokenizer.convert_tokens_to_ids('(*'),
                      my_tokenizer.convert_tokens_to_ids('▁(*')]


    prompt_tokens = my_tokenizer([prompt], return_tensors="pt")
    prompt_length = len(prompt_tokens.input_ids[0])
    tmp = model(**prompt_tokens, use_cache=True)
    cache = ((), tmp.past_key_values)
    logits[()] = process_logits(tmp.logits[0,-1,:],temperature=temperature)

    stack = []

    removed_probs : List[float] = []

    continue_p = 1.0

    # when the model looks up a definition that doesn't exist and we figure out what it meant
    # we put it in here.
    fake_env = {}

    # if you keep sampling after you've seen 99.9% of the sample space
    # you start getting really terrible, long generations.
    # such as the entire apache 2 license text.
    while sum(removed_probs)<0.999 and \
        (len(removed_probs)<10 or \
         (continue_p := unseen_test(removed_probs,0.05)) > sample_thresh):
        if tuple(stack) not in logits:
            prefix_length = 0
            for x,y in zip(cache[0],stack):
                if x == y:
                    prefix_length += 1
                else:
                    break

            if prefix_length == len(stack):
                print("really weird at", my_tokenizer.decode(stack), stack)
                prefix_length -= 1


            tmp = model(torch.tensor(
                [stack[prefix_length:]]),
                use_cache=True,
                past_key_values=trim_kvs(cache[1],
                                         prefix_length+prompt_length))

            logits[tuple(stack)] = process_logits(tmp.logits[0,-1,:],temperature=temperature)

            cache = (tuple(stack), tmp.past_key_values)

        p = logits[tuple(stack)]
        for x in COMMENT_TOKENS:
            # do NOT open comments!
            p[x] = 0.0

        stack.append(one_random_choice(
            population=list(p.keys()),
            weights=p.values()))

        # model just finished typing the name+type
        # of a theorem in the environment that we couldn't find.
        # probably not a real theorem. we'll help it out.
        if len(stack)>0 and stack[-1] == UNLOOKUP_TOKEN:
            print("attempting search")
            try:

                theorem = stack[-list(reversed(stack)).index(LOOKUP_TOKEN):-1]
                theorem_name = my_tokenizer.decode(theorem[:theorem.index(COLON_TOKEN)]).strip()
                theorem_type = my_tokenizer.decode(theorem[theorem.index(COLON_TOKEN)+1:]).strip()
            except ValueError:
                print("malformed search...")
                continue

            # ???
            if theorem_name in fake_env:
                continue


            search = f"Theorem {theorem_name} : {theorem_type}."

            # OK, now we just.... actually search

            with get_search_model(model) as search_model:
                vec = goodinference.embed(search_model,my_tokenizer,[search])[0].cuda().half()


            result = torch.argmax((vecs@vec).flatten())

            print(search,index[result])

            fake_env[theorem_name] = index[result]

            env[theorem_name] = index[result][1]

            # ok, now we'll just forget the type we made up and pretend it was a real thing that
            # always existed in the environment. we fall directly into type resolution below.

            stack = stack[:-list(reversed(stack)).index(LOOKUP_TOKEN)+theorem.index(COLON_TOKEN)+1]





        # if the model is typing the name of a thing that really exists in the environment,
        # go ahead and supply the type for it.
        if len(stack)>0 and stack[-1] == COLON_TOKEN:
            if stack.count(LOOKUP_TOKEN)>stack.count(UNLOOKUP_TOKEN):

                theorem_name = my_tokenizer.decode(
                    stack[-list(reversed(stack)).index(LOOKUP_TOKEN):-1])

                if theorem_name in env:
                    print("resolved name", theorem_name)
                    ty = re.sub(r"(\s)\s+"," ",env[theorem_name].strip())
                    new_toks = my_tokenizer(ty,add_special_tokens=False).input_ids + \
                        [UNLOOKUP_TOKEN]

                    for tok in new_toks:
                        probs = cast(DefaultDict[ProbsKey,ProbsValue],defaultdict(lambda:0))
                        probs[cast(ProbsKey,tok)] = cast(ProbsValue,1.0)
                        logits[tuple(stack)] = probs
                        stack.append(tok)



        if len(stack)>maxlength or \
            (my_tokenizer.decode(stack).endswith("\n```") and \
             len(stack)>0):

            if len(stack) <= maxlength:
                tactic_string = my_tokenizer.decode(stack)[:-4]
                # process lookups
                replacements = {}
                for my_match in re.finditer(r"<LOOKUP>\s*([^\s]+) : .+?<\/LOOKUP>", tactic_string):
                    theorem_name = my_match.group(1)
                    if theorem_name in fake_env:
                        theorem_name = fake_env[theorem_name][0]
                    replacements[m.group(0)] = theorem_name
                #pylint:disable=consider-using-dict-items
                for r in replacements:
                    tactic_string = tactic_string.replace(r,replacements[r])
                    print("running replace",r,replacements[r])
            prob = 1.0
            for i in range(len(stack)):
                prob *= logits[tuple(stack[:-i-1])][stack[-i-1]]
                logits[tuple(stack[:-i-1])][stack[-i-1]] -= prob
            removed_probs.append(prob)
            print(f"continue_p at {continue_p}")
            overloaded = len(stack) > maxlength
            stack = []
            for x in fake_env:
                del env[x]
            fake_env = {}
            if not overloaded:
                # when overloaded only then is tactic_string unbound
                tactic_string = cast(str,tactic_string)
                print(tactic_string,prob)
                yield tactic_string, prob
    #pylint:disable=pointless-string-statement
    """
    Stopped the while loop because either
    - sum(removed_probs)>=.999
    - len(removed_probs)>=10 and the unseen test gave something below threshold
    """


type ChunkType = str

def tokenize_glb_chunks(tok: TokenizerType,
                        chunks : Iterable[ChunkType],max_acc : int | float) -> Iterable[ChunkType]:
    """
    return `i` such that tokenizing `chunks[:i]` will not exceed `max_acc` tokens
    """
    acc = 0
    for i,c in enumerate(chunks):
        acc += len(tok(c).input_ids)
        if acc >= max_acc:
            return chunks[:i]
    return chunks


def mkprompt(tok : TokenizerType,
             diff : str,
             proof_history : List[ChunkType],
             proof_pane : Optional[StacksValues],
             budget : int =2048) -> Optional[str]:
    """
    TODO
    """
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

m = AutoModelForCausalLM.from_pretrained(
    "tomreichel/llemma-7b-extratok",
    device_map="auto",
    use_cache=False,
    quantization_config=bnb_config)

# do a little dance to load all the adapters
m.load_adapter("tomreichel/proofdb-HN-CLM","search")
m.disable_adapters()
m.load_adapter("tomreichel/proof-repair-model","tactic")
m.disable_adapters()
m.enable_adapters()
m.set_adapter("tactic")

tokenizer = AutoTokenizer.from_pretrained("tomreichel/repair-tokenizer")

type TacticType = Any
type ProbType = Any

async def filter_tactics(stack_manager : StackManager,
                         stack : List[StackManagerEvaluatesItem] | StacksKeys,
                         future : List[str],
                         tactics : Iterable[Tuple[TacticType,ProbType]]) -> \
                            Dict[TacticType,ProbType]:
    """
    TODO
    """

    attempts : List[Tuple[TacticType,ProbType]] = []
    for tactic,prob in tactics:
        if len(tactic) == 0:
            continue

        if tactic[0].isupper():
            continue

        if tactic.strip()[-1] != ".":
            if len(tactic)>1:
                # not a bullet and not a sentence??
                continue

        if "(*" in tactic or "*)" in tactic:
            continue

        if tactic[0].islower():
            attempt = await stack_manager.evaluate(stack + [f"progress {tactic}"])
        else:
            # can't progress a bullet
            attempt = await stack_manager.evaluate(stack + [tactic])

        if attempt is not None:
            attempts.append((tactic,prob))


    # evaluate attempts
    future_scores : Dict[TacticType,int] = {}
    for tactic,prob in attempts:
        future_score = 0
        for j in range(1,len(future)+1):
            res = await stack_manager.evaluate(stack+[tactic]+future[:j])
            if res is not None:
                future_score += 1
            else:
                break

        future_scores[tactic] = future_score

    print(future_scores)
    if len(attempts)>0:
        max_future = max(future_scores.values())

        # prune anything that doesn't have the maximal future score we found.
        # it CAUSES some kind of detectable breakage!
        attempts : Dict[TacticType,ProbType] = {
            tactic:prob for tactic,prob in attempts
            if future_scores[tactic] == max_future}

        # sum up probability we're left with
        prob_mass = sum(attempts.values())

        # divide out
        attempts = {tactic:prob/prob_mass for tactic,prob in attempts.items()}

        print(attempts)

        return attempts
    return {}

type RepairProofReturn = Optional[List[StackManagerEvaluatesItem]]

#pylint:disable=too-many-branches,too-many-statements
async def repair_proof(sentences : List[CoqSentence],
                       proof_start : int,
                       proof_end : int,
                       diff : str,flags : str,
                       gpu_lock : aio.Lock,
                       _recommendations=None) -> RepairProofReturn:
    """
    TODO
    """
    coq = CoqProcess(*flags.split())
    _stdout = await coq.run("\n".join(x.text for x in sentences[:proof_start]), return_stderr=False)
    env = await coq.environment()
    coq.close()

    stack_manager = StackManager([x.text for x in sentences[:proof_start]], flags)

    completion_cache : Dict[tuple,Dict[TacticType,ProbType]] = {}

    # assuming we won't more-than-double proof length
    stack : List[StackManagerEvaluatesItem] = []

    _state = await stack_manager.evaluate([])

    while True:

        print(stack)

        test = await stack_manager.evaluate(stack + ["Qed."])
        if test is not None:
            print("we did it!")
            print(test)
            return stack
            # break
        print("not done yet.")


        alignment = cast(Alignment,
                         prefix_alignment(
                             [x.text for x in sentences[proof_start:proof_end]],
                             stack,
                             return_cost = False))

        old_cnt = 0
        proof_history = []
        for _j,(x,y) in enumerate(alignment):
            if x is None:
                proof_history.append(f"+  {y}")
                continue
            old_cnt += 1 # used a stmt from old proof
            if y is None:
                proof_history.append(f"-  {x}")
                continue
            if x==y:
                proof_history.append(f"   {x}")
            else:
                proof_history.append(f"-  {x}")
                proof_history.append(f"+  {y}")

        if ((old_cnt == proof_end-proof_start and all(x is None for x,y in alignment[-8:]))
            or (tuple(stack) in completion_cache and \
                (sum(completion_cache[tuple(stack)].values()) == 0))):
            # we're either falling off the end of the old proof and we haven't solved it
            # or we tried to find tactics to continue and utterly failed.
            # backtracking time.

            if len(stack) == 0:
                break

            # ensure we don't do this again.
            mass = 1.0
            for i in range(len(stack)-1,-1,-1):
                cur_stack = tuple(stack[:i])
                mass *= completion_cache[cur_stack][stack[i]]
                completion_cache[cur_stack][stack[i]] -= mass


            # aaand reset
            stack = []
            print("trying again...")
            continue


        recommendation = sentences[old_cnt+proof_start].text
        for s in sentences[old_cnt+proof_start:old_cnt+proof_start+3]:
            proof_history.append(f"?  {s.text}")


        if (await stack_manager.evaluate(stack+[recommendation])) is not None:
            # ok, it looks like the original proof still works here.
            # we'll just take that line
            completion_cache[tuple(stack)] = {sentences[old_cnt+proof_start].text:1.0}
            stack.append(sentences[old_cnt+proof_start].text)
            continue


        proof_pane = await stack_manager.evaluate(stack)

        print(proof_pane)

        if proof_pane is None:
            completion_cache[tuple(stack)] = {}


        if tuple(stack) not in completion_cache:

            # unfortunately, the fast tokenizer implementation
            # is not thread safe.
            async with gpu_lock:
                prompt = mkprompt(tokenizer,diff,proof_history,proof_pane)
            print(prompt)

            if prompt is None:
                # too big to work with?!
                completion_cache[tuple(stack)] = {}
                continue

            s = sample(m,tokenizer,prompt,env,p=0.1)

            async with gpu_lock:
                #attempts = await aio.to_thread(list,s)
                attempts = list(s)

            print(attempts)

            future = [x.text for x in sentences[old_cnt+proof_start+1:proof_end]] + ["Qed."]

            completion_cache[tuple(stack)] = \
                await filter_tactics(stack_manager,stack,future,attempts)


        probs = completion_cache[tuple(stack)]
        if len(probs) > 0 and sum(probs.values()) > 0:
            stack.append(one_random_choice(
                population=list(probs.keys()),
                weights=probs.values()
            ))

    return None
