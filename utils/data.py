import os, pickle, copy, sys, copy, json, itertools
import torch
from utils.alphabet import Alphabet
from transformers import BertTokenizer, BertTokenizerFast, AlbertTokenizer

def list_index(list1: list, list2: list) -> list:
    start = [i for i, x in enumerate(list2) if x == list1[0]]
    end = [i for i, x in enumerate(list2) if x == list1[-1]]
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]
    else:
        for i in start:
            for j in end:
                if i <= j:
                    if list2[i:j+1] == list1:
                        index = (i, j)
                        break
        return index[0], index[1]
        

def child_data_process(input_doc, relational_alphabet, entity_type_alphabet, tokenizer, evaluate, repeat_gt_entities=-1, repeat_gt_triples=-1, max_len=400, max_ent=80):
    samples = []
    total_triples = 0
    total_entities = 0
    max_triples = 0
    max_entities = 0
    max_len_mention = 0
    num_samples = 0
    avg_len = 0
    print(input_doc)

    avg_no_entity_len = 0
    num_no_entity = 0

    with open(input_doc) as f:
        lines = json.load(f)

    for idx, line in enumerate(lines):
        doc_key = line["doc_key"]
        sent_id = line["sent_id"]
        ori_pos = line['original_position'] if 'original_position' in line else None
        idx = doc_key + '.' + str(sent_id)

        text = line["text"]
        enc = tokenizer(text, add_special_tokens=True)
        sent_id = enc['input_ids']

        if len(line['entities']) == 0:
        # if not evaluate and len(line['entities']) == 0:
            avg_no_entity_len += len(sent_id)
            num_no_entity += 1
            continue

        # if len(line['entities']) <= 1:
        #     continue

        if len(line['entities']) >= max_ent:
            max_entities = max(max_entities, len(line['entities']))
            continue

        if len(sent_id) > max_len:
            continue

        avg_len += len(text)

        sent_seg_encoding = [0] * len(sent_id)
        context2token_masks = None
        token_masks = [1] * len(sent_id)

        char_to_bep = dict()
        bep_to_char = dict()
        for i in range(len(text)):
            bep_index = enc.char_to_token(i)
            char_to_bep[i] = bep_index
            if bep_index in bep_to_char:
                left, right = bep_to_char[bep_index][0], bep_to_char[bep_index][-1]
                bep_to_char[bep_index] = [left, max(right, i)]
            else:
                bep_to_char[bep_index] = [i, i]
        
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": [],
                    "head_mention": [], "tail_mention": [], 'head_part_labels': [], 'tail_part_labels': [], 'head_type': [], 'tail_type': [],
                    "head_entID": [], "tail_entID": [], "rel_entID_labels": [],
                    "ent_type": [], "ent_start_index": [], "ent_end_index": [], "ent_part_labels": [], 'ent_have_rel': [],
                    "relID_labels": [], "relID_head_labels": [], "relID_tail_labels": []
                }

        flag_invalid = False
        if evaluate:
            triples = line["relations"]
            for triple in triples:
                relation_id = relational_alphabet.get_index(triple[-1])
                h_mention = triple[0][2]
                t_mention = triple[1][2]

                max_len_mention = max(max_len_mention, len(h_mention))
                max_len_mention = max(max_len_mention, len(t_mention))

                target["relation"].append(relation_id)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)

            entities = line["entities"]
            for ent in entities:
                ent_type_id = entity_type_alphabet.get_index(ent[-1])
                ent_start, ent_end = ent[0], ent[1]
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end-1]
                if ent_start_index is None or ent_end_index is None:
                    flag_invalid = True
                    continue

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)

            if flag_invalid:
                continue
            
            samples.append([idx, sent_id, target, None, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, None, None, ori_pos])

            total_triples += len(triples)
            total_entities += len(entities)
            max_triples = max(max_triples, len(triples))
            max_entities = max(max_entities, len(entities))
            num_samples += 1

        else:
            repeat_num = 1
            triples = line["relations"]
            entities = line["entities"]

            set_head_tail = set()

            span2entID = {}
            span2etype = {}
            for entID, ent in enumerate(entities):
                span2entID[tuple(ent[:2])] = entID
                span2etype[tuple(ent[:2])] = ent[-1]

            span2relID_head = {}
            span2relID_tail = {}
            span2relID = {}
            for span in span2entID.keys():
                span2relID_head[span] = [0] * len(triples)
                span2relID_tail[span] = [0] * len(triples)
                span2relID[span] = [0] * len(triples)
                for relID, triple in enumerate(triples):
                    if span == triple[0][:2]:
                        span2relID_head[span][relID] = 1
                        span2relID[span][relID] = 1
                    if span == triple[1][:2]:
                        span2relID_tail[span][relID] = 1
                        span2relID[span][relID] = 1


            triples_ = triples
            relID2triples = dict()
            if repeat_gt_triples != -1 and len(triples) > 0:
                repeat_gt_triples_ = max(repeat_gt_triples, len(triples))
                k = repeat_gt_triples_ // len(triples)
                m = repeat_gt_triples_ % len(triples)
                triples_ = triples * k

                relID2triples = dict([(i, [len(triples)*j + i for j in range(k)])
                                    for i in range(len(triples))])

            for triple in triples_:
                head, tail, rel_type = triple[0], triple[1], triple[2]
                relation_id = relational_alphabet.get_index(rel_type)
                h_start, h_end, h_mention, h_label = head
                t_start, t_end, t_mention, t_label = tail

                assert text[h_start: h_end] == h_mention
                assert text[t_start: t_end] == t_mention
                assert h_label == span2etype[(h_start, h_end)]
                assert t_label == span2etype[(t_start, t_end)]

                try:
                    head_start_index, head_end_index = char_to_bep[h_start], char_to_bep[h_end-1]
                    tail_start_index, tail_end_index = char_to_bep[t_start], char_to_bep[t_end-1]
                except:
                    print(line['doc_key'])
                    print(tokenizer.tokenize(text))
                    print(char_to_bep)
                if head_start_index is None or head_end_index is None or tail_start_index is None or tail_end_index is None:
                    flag_invalid = True
                    continue

                max_len_mention = max(max_len_mention, h_end-h_start+1)
                max_len_mention = max(max_len_mention, t_end-t_start+1)

                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(h_label))
                target["tail_type"].append(entity_type_alphabet.get_index(t_label))

                h_entID = span2entID[(h_start, h_end)]
                t_entID = span2entID[(t_start, t_end)]
                target["head_entID"].append(h_entID)
                target["tail_entID"].append(t_entID)

                rel_entID_labels = [0] * len(entities)
                rel_entID_labels[h_entID] = 1
                rel_entID_labels[t_entID] = 1
                target["rel_entID_labels"].append(rel_entID_labels)

                head_part_labels = [0.0] * len(sent_id)
                for index in range(head_start_index, head_end_index+1):
                    head_part_labels[index] = 0.5
                head_part_labels[head_start_index] = 1.0
                head_part_labels[head_end_index] = 1.0
                target["head_part_labels"].append(head_part_labels)
                
                tail_part_labels = [0.0] * len(sent_id)
                for index in range(tail_start_index, tail_end_index+1):
                    tail_part_labels[index] = 0.5
                tail_part_labels[tail_start_index] = 1.0
                tail_part_labels[tail_end_index] = 1.0
                target["tail_part_labels"].append(tail_part_labels)

                set_head_tail.add(tuple(head[:2]))
                set_head_tail.add(tuple(tail[:2]))

            bio_labels = [0] * len(sent_id)

            entities_ = entities
            entID2entities = dict()
            if repeat_gt_entities != -1 and len(entities) > 0:
                repeat_gt_entities_ = max(repeat_gt_entities, len(entities))
                k = repeat_gt_entities_ // len(entities)
                m = repeat_gt_entities_ % len(entities)
                entities_ = entities * k
                entities_ += entities[:m]

                entID2entities = dict([(i, [len(entities)*j + i for j in range(k)])
                                    for i in range(len(entities))])
                for entID in entID2entities:
                    if entID < m:
                        entID2entities[entID].append(len(entities)*k + entID)

            for ent in entities_:
                ent_type = ent[-1]

                ent_type_id = entity_type_alphabet.get_index(ent_type)
                ent_start, ent_end = ent[0], ent[1]
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end-1]
                if ent_start_index is None or ent_end_index is None:
                    flag_invalid = True
                    continue

                if text[ent_start: ent_end] != ent[2]:
                    print('text[ent_start: ent_end] != ent[2]')
                    print(line)
                    exit(0)

                max_len_mention = max(max_len_mention, ent_end-ent_start+1)

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)
                target["relID_labels"].append(span2relID[tuple(ent[:2])])
                target["relID_head_labels"].append(span2relID_head[tuple(ent[:2])])
                target["relID_tail_labels"].append(span2relID_tail[tuple(ent[:2])])

                ent_have_rel = 1 if tuple(ent[:2]) in set_head_tail else 0
                target["ent_have_rel"].append(ent_have_rel)

                bio_labels[ent_start_index] = 1
                for index in range(ent_start_index+1, ent_end_index+1):
                    bio_labels[index] = 2

                ent_part_labels = [0.0] * len(sent_id)
                for index in range(ent_start_index, ent_end_index+1):
                    ent_part_labels[index] = 0.5
                ent_part_labels[ent_start_index] = 1.0
                ent_part_labels[ent_end_index] = 1.0
                target["ent_part_labels"].append(ent_part_labels)

            if flag_invalid:
                continue
            for _ in range(repeat_num):
                samples.append([idx, sent_id, target, bio_labels, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, entID2entities, relID2triples, ori_pos])

            total_triples += len(triples)
            total_entities += len(entities)
            max_triples = max(max_triples, len(triples))
            max_entities = max(max_entities, len(entities))
            num_samples += 1

    print('[num samples]:', num_samples)
    print('[avg len]:', avg_len / num_samples)
    print('[num triples]:', total_triples)
    print('[avg triples]:', total_triples / num_samples)
    print('[max triples]:', max_triples)
    print('[num entities]:', total_entities)
    print('[avg entities]:', total_entities / num_samples)
    print('[max entities]:', max_entities)
    print('[max len mention]:', max_len_mention)
    print('[# samples without entity]:', num_no_entity)
    print()

    return samples

def text2dt_data_process(input_doc, relational_alphabet, entity_type_alphabet, tokenizer, evaluate, repeat_gt_entities=-1, repeat_gt_triples=-1):
    samples = []
    total_triples = 0
    max_triples = 0
    max_entities = 0
    max_len_mention = 0
    num_samples = 0
    print(input_doc)

    with open(input_doc) as f:
        lines = f.readlines()
        lines = [eval(ele) for ele in lines]
    for idx, line in enumerate(lines):
        text = line["text"]
        enc = tokenizer(text, add_special_tokens=True)
        sent_id = enc['input_ids']

        sent_seg_encoding = [0] * len(sent_id)
        context2token_masks = None
        # token_masks = [1] * (len(sent_id) - 2)
        token_masks = [1] * len(sent_id)

        char_to_bep = dict()
        bep_to_char = dict()
        for i in range(len(text)):
            bep_index = enc.char_to_token(i)
            char_to_bep[i] = bep_index
            if bep_index in bep_to_char:
                left, right = bep_to_char[bep_index][0], bep_to_char[bep_index][-1]
                bep_to_char[bep_index] = [left, max(right, i)]
            else:
                bep_to_char[bep_index] = [i, i]
        
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": [],
                    "head_mention": [], "tail_mention": [], 'head_part_labels': [], 'tail_part_labels': [], 'head_type': [], 'tail_type': [],
                    "head_entID": [], "tail_entID": [], "rel_entID_labels": [],
                    "ent_type": [], "ent_start_index": [], "ent_end_index": [], "ent_part_labels": [], 'ent_have_rel': [],
                    "relID_labels": [], "relID_head_labels": [], "relID_tail_labels": []
                }

        if evaluate:
            triples = line["relations"]
            for triple in triples:
                relation_id = relational_alphabet.get_index(triple[1])
                h_mention = triple[0][-1]
                t_mention = triple[2][-1]

                max_len_mention = max(max_len_mention, len(h_mention))
                max_len_mention = max(max_len_mention, len(t_mention))

                target["relation"].append(relation_id)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)

            # entities = line["entities"]
            # for ent in entities:
            #     # ent_type_id = entity_type_alphabet.get_index(ent[2])
            #     ent_type_id = entity_type_alphabet.get_index('ENTITY')
            #     ent_start, ent_end = ent[0], ent[1]
            #     ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end-1]

            #     target["ent_type"].append(ent_type_id)
            #     target["ent_start_index"].append(ent_start_index)
            #     target["ent_end_index"].append(ent_end_index)

            samples.append([idx, sent_id, target, None, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, None, None])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            num_samples += 1

        else:
            repeat_num = 1
            triples = line["relations"]
            entities = line["entities"]

            set_head_tail = set()

            span2entID = {}
            span2etype = {}
            for entID, ent in enumerate(entities):
                span2entID[ent[:2]] = entID
                span2etype[ent[:2]] = ent[-1]

            span2relID_head = {}
            span2relID_tail = {}
            span2relID = {}
            for span in span2entID.keys():
                span2relID_head[span] = [0] * len(triples)
                span2relID_tail[span] = [0] * len(triples)
                span2relID[span] = [0] * len(triples)
                for relID, triple in enumerate(triples):
                    if span == triple[0][:2]:
                        span2relID_head[span][relID] = 1
                        span2relID[span][relID] = 1
                    if span == triple[1][:2]:
                        span2relID_tail[span][relID] = 1
                        span2relID[span][relID] = 1


            triples_ = triples
            relID2triples = dict()
            if repeat_gt_triples != -1 and len(triples) > 0:
                # unfixed
                repeat_gt_triples = max(repeat_gt_triples, len(triples))
                k = repeat_gt_triples // len(triples)
                m = repeat_gt_triples % len(triples)
                triples_ = triples * k

                relID2triples = dict([(i, [len(triples)*j + i for j in range(k)])
                                    for i in range(len(triples))])

            for triple in triples_:
                head, rel_type, tail = triple[0], triple[1], triple[2]
                relation_id = relational_alphabet.get_index(rel_type)
                h_start, h_end, h_mention = head
                t_start, t_end, t_mention = tail

                assert text[h_start: h_end] == h_mention
                assert text[t_start: t_end] == t_mention

                try:
                    head_start_index, head_end_index = char_to_bep[h_start], char_to_bep[h_end-1]
                    tail_start_index, tail_end_index = char_to_bep[t_start], char_to_bep[t_end-1]
                except:
                    print(line['fname'])
                    print(tokenizer.tokenize(text))
                    print(char_to_bep)

                max_len_mention = max(max_len_mention, h_end-h_start+1)
                max_len_mention = max(max_len_mention, t_end-t_start+1)

                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(span2etype[(h_start, h_end)]))
                target["tail_type"].append(entity_type_alphabet.get_index(span2etype[(t_start, t_end)]))

                h_entID = span2entID[(h_start, h_end)]
                t_entID = span2entID[(t_start, t_end)]
                target["head_entID"].append(h_entID)
                target["tail_entID"].append(t_entID)

                rel_entID_labels = [0] * len(entities)
                rel_entID_labels[h_entID] = 1
                rel_entID_labels[t_entID] = 1
                target["rel_entID_labels"].append(rel_entID_labels)

                head_part_labels = [0.0] * len(sent_id)
                for index in range(head_start_index, head_end_index+1):
                    head_part_labels[index] = 0.5
                head_part_labels[head_start_index] = 1.0
                head_part_labels[head_end_index] = 1.0
                target["head_part_labels"].append(head_part_labels)
                
                tail_part_labels = [0.0] * len(sent_id)
                for index in range(tail_start_index, tail_end_index+1):
                    tail_part_labels[index] = 0.5
                tail_part_labels[tail_start_index] = 1.0
                tail_part_labels[tail_end_index] = 1.0
                target["tail_part_labels"].append(tail_part_labels)

                set_head_tail.add(tuple(head[:2]))
                set_head_tail.add(tuple(tail[:2]))

            bio_labels = [0] * len(sent_id)

            entities_ = entities
            entID2entities = dict()
            if repeat_gt_entities != -1 and len(entities) > 0:
                repeat_gt_entities_ = max(repeat_gt_entities, len(entities))
                k = repeat_gt_entities_ // len(entities)
                m = repeat_gt_entities_ % len(entities)
                entities_ = entities * k
                entities_ += entities[:m]

                entID2entities = dict([(i, [len(entities)*j + i for j in range(k)])
                                    for i in range(len(entities))])
                for entID in entID2entities:
                    if entID < m:
                        entID2entities[entID].append(len(entities)*k + entID)

            for ent in entities_:
                ent_type = 'ENTITY' if len(ent) == 3 else ent[-1]

                ent_type_id = entity_type_alphabet.get_index(ent_type)
                ent_start, ent_end = ent[0], ent[1]
                ent_start_index, ent_end_index = char_to_bep[ent_start], char_to_bep[ent_end-1]

                if text[ent_start: ent_end] != ent[2]:
                    print(line)

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)
                target["relID_labels"].append(span2relID[tuple(ent[:2])])
                target["relID_head_labels"].append(span2relID_head[tuple(ent[:2])])
                target["relID_tail_labels"].append(span2relID_tail[tuple(ent[:2])])

                ent_have_rel = 1 if ent[:2] in set_head_tail else 0
                target["ent_have_rel"].append(ent_have_rel)

                bio_labels[ent_start_index] = 1
                for index in range(ent_start_index+1, ent_end_index+1):
                    bio_labels[index] = 2

                ent_part_labels = [0.0] * len(sent_id)
                for index in range(ent_start_index, ent_end_index+1):
                    ent_part_labels[index] = 0.5
                ent_part_labels[ent_start_index] = 1.0
                ent_part_labels[ent_end_index] = 1.0
                target["ent_part_labels"].append(ent_part_labels)

            for _ in range(repeat_num):
                samples.append([idx, sent_id, target, bio_labels, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, entID2entities, relID2triples])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            max_entities = max(max_entities, len(entities))
            num_samples += 1

    print('[num samples]:', num_samples)
    print('[avg triples]:', total_triples / num_samples)
    print('[max triples]:', max_triples)
    if 'test' not in input_doc:
        print('[max entities]:', max_entities)
    print('[max len mention]:', max_len_mention)
    print()

    return samples
    

def biorelex_data_process(input_doc, relational_alphabet, entity_type_alphabet, tokenizer, evaluate, repeat_gt_entities=-1, repeat_gt_triples=-1):
    # Read raw instances
    raw_insts = []
    print(input_doc)
    with open(input_doc, 'r', encoding='utf-8') as f:
        raw_insts += json.load(f)

    # Construct data_insts
    samples = []
    total_triples = 0
    max_triples = 0
    total_entities = 0
    max_entities = 0
    max_len_mention = 0
    num_samples = 0

    for inst in raw_insts:
        idx, text = inst['id'], inst['text']
        raw_entites, raw_interactions = inst.get('entities', []), inst.get('interactions', [])

        enc = tokenizer(text, add_special_tokens=True)
        sent_id = enc['input_ids']
        # print(tokenizer.convert_ids_to_tokens(sent_id))

        sent_seg_encoding = [0] * len(sent_id)
        context2token_masks = None
        token_masks = [1] * len(sent_id)

        char_to_bep = dict()
        bep_to_char = dict()
        for i in range(len(text)):
            bep_index = enc.char_to_token(i)
            char_to_bep[i] = bep_index
            if bep_index in bep_to_char:
                left, right = bep_to_char[bep_index][0], bep_to_char[bep_index][-1]
                assert right < i
                bep_to_char[bep_index] = [left, i]
            else:
                bep_to_char[bep_index] = [i, i]

        # Compute entities
        entities = []
        eid2mentions = dict()
        if len(raw_entites) > 0:
            appeared = set()
            for eid, e in enumerate(raw_entites):
                # if len(e['names']) > 1:
                #     print(e['names'])
                ent_type = e['label']
                for name in e['names']:
                    for start, end in e['names'][name]['mentions']:
                        if (start, end) in appeared: continue
                        appeared.add((start, end))

                        start_index, end_index = char_to_bep[start], char_to_bep[end-1]
                        mention = (start_index, end_index, name, ent_type)
                        entities.append(mention)

                        len_mention = end_index - start_index + 1
                        max_len_mention = max(max_len_mention, len_mention)

                        if eid not in eid2mentions:
                            eid2mentions[eid] = []
                        eid2mentions[eid].append(mention)

        # Compute relations
        triples = []
        if len(raw_interactions) > 0:
            for interaction in raw_interactions:
                p1, p2 = interaction['participants']
                label = interaction['label']

                if evaluate:
                    for e1 in eid2mentions[p1]:
                        for e2 in eid2mentions[p2]:
                            triples.append((e1, e2, label))
                else:
                    distance2triples = dict()
                    min_distance = 10000

                    for e1 in eid2mentions[p1]:
                        for e2 in eid2mentions[p2]:
                            distance = abs(e1[0] + e1[1] - e2[0] - e2[1])
                            min_distance = min(min_distance, distance)
                            if distance not in distance2triples:
                                distance2triples[distance] = []
                            distance2triples[distance].append((e1, e2, label))


                    for triple in distance2triples[min_distance]:
                        if triple not in triples:
                            triples.append(triple)

                    for distance, triples_ in distance2triples.items():
                        if distance > 30:
                            continue
                        for triple in triples_:
                            if triple not in triples:
                                triples.append(triple)

        # print(triples)
        # print()
        
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": [],
                    "head_mention": [], "tail_mention": [], 'head_part_labels': [], 'tail_part_labels': [], 'head_type': [], 'tail_type': [],
                    "head_entID": [], "tail_entID": [], "rel_entID_labels": [],
                    "ent_type": [], "ent_start_index": [], "ent_end_index": [], "ent_part_labels": [], 'ent_have_rel': [],
                    "relID_labels": [], "relID_head_labels": [], "relID_tail_labels": []
                }

        if evaluate:
            for triple in triples:
                relation_id = relational_alphabet.get_index(triple[2])
                h_mention = triple[0][2]
                t_mention = triple[1][2]

                target["relation"].append(relation_id)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)

            for ent in entities:
                ent_type_id = entity_type_alphabet.get_index(ent[3])
                # ent_type_id = entity_type_alphabet.get_index('ENTITY')
                ent_start_index, ent_end_index = ent[0], ent[1]

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)

            samples.append([idx, sent_id, target, None, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, inst, None, None])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            total_entities += len(entities)
            max_entities = max(max_entities, len(entities))
            num_samples += 1
        
        else:
            span2entID = {}
            for entID, ent in enumerate(entities):
                span2entID[tuple(ent[:2])] = entID

            span2relID_head = {}
            span2relID_tail = {}
            span2relID = {}
            for span in span2entID.keys():
                span2relID_head[span] = [0] * len(triples)
                span2relID_tail[span] = [0] * len(triples)
                span2relID[span] = [0] * len(triples)
                for relID, triple in enumerate(triples):
                    if span == triple[0][:2]:
                        span2relID_head[span][relID] = 1
                        span2relID[span][relID] = 1
                    if span == triple[1][:2]:
                        span2relID_tail[span][relID] = 1
                        span2relID[span][relID] = 1

            repeat_num = 1
            set_head_tail = set()

            inverse_triples = []
            for triple in triples:
                inverse_triple = (triple[1], triple[0], triple[2])
                if inverse_triple in triples:
                    # print('idx', idx)
                    # print(triple)
                    continue
                inverse_triples.append(inverse_triple)
            triples += inverse_triples

            triples_ = triples
            relID2triples = dict()
            if repeat_gt_triples != -1 and len(triples) > 0:
                repeat_gt_triples_ = max(repeat_gt_triples, len(triples))
                k = repeat_gt_triples_ // len(triples)
                m = repeat_gt_triples_ % len(triples)
                triples_ = triples * k

                relID2triples = dict([(i, [len(triples)*j + i for j in range(k)])
                                    for i in range(len(triples))])

            for triple in triples_:
                head, tail, rel_type = triple[0], triple[1], triple[2]
                relation_id = relational_alphabet.get_index(rel_type)
                head_start_index, head_end_index, h_mention, h_type = head
                tail_start_index, tail_end_index, t_mention, t_type = tail

                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(h_type))
                target["tail_type"].append(entity_type_alphabet.get_index(t_type))

                h_entID = span2entID[(head_start_index, head_end_index)]
                t_entID = span2entID[(tail_start_index, tail_end_index)]
                target["head_entID"].append(h_entID)
                target["tail_entID"].append(t_entID)

                rel_entID_labels = [0] * len(entities)
                rel_entID_labels[h_entID] = 1
                rel_entID_labels[t_entID] = 1
                target["rel_entID_labels"].append(rel_entID_labels)

                head_part_labels = [0.0] * len(sent_id)
                for index in range(head_start_index, head_end_index+1):
                    head_part_labels[index] = 0.5
                head_part_labels[head_start_index] = 1.0
                head_part_labels[head_end_index] = 1.0
                target["head_part_labels"].append(head_part_labels)
                
                tail_part_labels = [0.0] * len(sent_id)
                for index in range(tail_start_index, tail_end_index+1):
                    tail_part_labels[index] = 0.5
                tail_part_labels[tail_start_index] = 1.0
                tail_part_labels[tail_end_index] = 1.0
                target["tail_part_labels"].append(tail_part_labels)

                set_head_tail.add(tuple(head[:2]))
                set_head_tail.add(tuple(tail[:2]))

            bio_labels = [0] * len(sent_id)

            entities_ = entities
            entID2entities = dict()
            if repeat_gt_entities != -1 and len(entities) > 0:
                repeat_gt_entities_ = max(repeat_gt_entities, len(entities))
                k = repeat_gt_entities_ // len(entities)
                m = repeat_gt_entities_ % len(entities)
                entities_ = entities * k
                entities_ += entities[:m]

                entID2entities = dict([(i, [len(entities)*j + i for j in range(k)])
                                    for i in range(len(entities))])
                for entID in entID2entities:
                    if entID < m:
                        entID2entities[entID].append(len(entities)*k + entID)

            for ent in entities_:
                ent_type = ent[-1]
                # ent_type = 'ENTITY'

                ent_type_id = entity_type_alphabet.get_index(ent_type)
                ent_start_index, ent_end_index = ent[0], ent[1]

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)
                target["relID_labels"].append(span2relID[tuple(ent[:2])])
                target["relID_head_labels"].append(span2relID_head[tuple(ent[:2])])
                target["relID_tail_labels"].append(span2relID_tail[tuple(ent[:2])])

                ent_have_rel = 1 if ent[:2] in set_head_tail else 0
                target["ent_have_rel"].append(ent_have_rel)

                bio_labels[ent_start_index] = 1
                for index in range(ent_start_index+1, ent_end_index+1):
                    bio_labels[index] = 2

                ent_part_labels = [0.0] * len(sent_id)
                for index in range(ent_start_index, ent_end_index+1):
                    ent_part_labels[index] = 0.5
                ent_part_labels[ent_start_index] = 1.0
                ent_part_labels[ent_end_index] = 1.0
                target["ent_part_labels"].append(ent_part_labels)

            for _ in range(repeat_num):
                samples.append([idx, sent_id, target, bio_labels, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, entID2entities, relID2triples])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            total_entities += len(entities)
            max_entities = max(max_entities, len(entities))
            num_samples += 1

    print('[num samples]:', num_samples)
    print('[avg triples]:', total_triples / num_samples)
    print('[max triples]:', max_triples)
    print('[avg entities]:', total_entities / num_samples)
    print('[max entities]:', max_entities)
    print('[max len mention]:', max_len_mention)
    print()

    return samples

def ade_data_process(input_doc, relational_alphabet, entity_type_alphabet, tokenizer, evaluate, repeat_gt_entities=-1, repeat_gt_triples=-1):
    # Read raw instances
    print(input_doc)
    with open(input_doc, 'r') as f:
        data = json.loads(f.read())

    # Construct data_insts
    samples = []
    total_triples = 0
    max_triples = 0
    total_entities = 0
    max_entities = 0
    max_len_mention = 0
    num_samples = 0
    for raw_inst in data:
        idx, tokens = raw_inst['orig_id'], raw_inst['tokens']
        raw_entities, raw_relations = raw_inst['entities'], raw_inst['relations']
        text = ' '.join(tokens)

        # Initialize inst_data
        inst_data = {
            'text': text, 'id': id, 'entities':[], 'interactions': []
        }

        # Compute mappings from original tokens to char offsets
        otoken2startchar, otoken2endchar, char_offset = {}, {}, 0
        for ix in range(len(tokens)):
            otoken2startchar[ix] = char_offset
            otoken2endchar[ix] = char_offset + len(tokens[ix])
            char_offset += (1 + len(tokens[ix]))

        enc = tokenizer(text, add_special_tokens=True)
        sent_id = enc['input_ids']
        # print(tokenizer.convert_ids_to_tokens(sent_id))

        sent_seg_encoding = [0] * len(sent_id)
        context2token_masks = None
        token_masks = [1] * len(sent_id)

        char_to_bep = dict()
        bep_to_char = dict()
        for i in range(len(text)):
            bep_index = enc.char_to_token(i)
            char_to_bep[i] = bep_index
            if bep_index in bep_to_char:
                left, right = bep_to_char[bep_index][0], bep_to_char[bep_index][-1]
                assert right < i
                bep_to_char[bep_index] = [left, i]
            else:
                bep_to_char[bep_index] = [i, i]

        # Compute entities
        entities = []
        if len(raw_entities) > 0:
            for e in raw_entities:
                ent_type = e['type']
                start, end = otoken2startchar[e['start']], otoken2endchar[e['end']-1]
                name = ' '.join(tokens[e['start']:e['end']])
                assert name == text[start: end]
                start_index, end_index = char_to_bep[start], char_to_bep[end-1]
                # print(name)
                # print(text[start: end])
                # print(tokenizer.convert_ids_to_tokens(sent_id[start_index: end_index+1]))
                entity = (start_index, end_index, name, ent_type)
                entities.append(entity)

                inst_data['entities'].append({
                    'label': e['type'],
                    'names': {
                        name: {
                            'is_mentioned': True,
                            'mentions': [[start, end]]
                        }
                    },
                    'is_mentioned': True
                })

        # Compute relations
        triples = []
        if len(raw_relations) > 0:
            for relation in raw_relations:
                p1, p2 = relation['head'], relation['tail']
                label = relation['type']
                triples.append((entities[p1], entities[p2], label))
                inst_data['interactions'].append({
                                  'participants': [p1, p2],
                                  'label': label
                                })

        # print(text)
        # print(triples)
        # print()

        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": [],
                    "head_mention": [], "tail_mention": [], 'head_part_labels': [], 'tail_part_labels': [], 'head_type': [], 'tail_type': [],
                    "head_entID": [], "tail_entID": [], "rel_entID_labels": [],
                    "ent_type": [], "ent_start_index": [], "ent_end_index": [], "ent_part_labels": [], 'ent_have_rel': [],
                    "relID_labels": [], "relID_head_labels": [], "relID_tail_labels": []
                }

        if evaluate:
            for triple in triples:
                relation_id = relational_alphabet.get_index(triple[2])
                h_mention = triple[0][2]
                t_mention = triple[1][2]
                
                target["relation"].append(relation_id)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)

            for ent in entities:
                ent_type_id = entity_type_alphabet.get_index(ent[3])
                # ent_type_id = entity_type_alphabet.get_index('ENTITY')
                ent_start_index, ent_end_index = ent[0], ent[1]

                max_len_mention = max(max_len_mention, ent_end_index - ent_start_index + 1)

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)

            samples.append([idx, sent_id, target, None, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, inst_data, None])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            total_entities += len(entities)
            max_entities = max(max_entities, len(entities))
            num_samples += 1
        
        else:
            span2entID = {}
            for entID, ent in enumerate(entities):
                span2entID[tuple(ent[:2])] = entID

            span2relID_head = {}
            span2relID_tail = {}
            span2relID = {}
            for span in span2entID.keys():
                span2relID_head[span] = [0] * len(triples)
                span2relID_tail[span] = [0] * len(triples)
                span2relID[span] = [0] * len(triples)
                for relID, triple in enumerate(triples):
                    if span == triple[0][:2]:
                        span2relID_head[span][relID] = 1
                        span2relID[span][relID] = 1
                    if span == triple[1][:2]:
                        span2relID_tail[span][relID] = 1
                        span2relID[span][relID] = 1
                        
            repeat_num = 1
            set_head_tail = set()

            triples_ = triples
            relID2triples = dict()
            if repeat_gt_triples != -1 and len(triples) > 0:
                repeat_gt_triples_ = max(repeat_gt_triples, len(triples))
                k = repeat_gt_triples_ // len(triples)
                m = repeat_gt_triples_ % len(triples)
                triples_ = triples * k

                relID2triples = dict([(i, [len(triples)*j + i for j in range(k)])
                                    for i in range(len(triples))])

            for triple in triples_:
                head, tail, rel_type = triple[0], triple[1], triple[2]
                relation_id = relational_alphabet.get_index(rel_type)
                head_start_index, head_end_index, h_mention, h_type = head
                tail_start_index, tail_end_index, t_mention, t_type = tail

                target["relation"].append(relation_id)
                target["head_start_index"].append(head_start_index)
                target["head_end_index"].append(head_end_index)
                target["tail_start_index"].append(tail_start_index)
                target["tail_end_index"].append(tail_end_index)
                target["head_mention"].append(h_mention)
                target["tail_mention"].append(t_mention)
                target["head_type"].append(entity_type_alphabet.get_index(h_type))
                target["tail_type"].append(entity_type_alphabet.get_index(t_type))

                h_entID = span2entID[(head_start_index, head_end_index)]
                t_entID = span2entID[(tail_start_index, tail_end_index)]
                target["head_entID"].append(h_entID)
                target["tail_entID"].append(t_entID)

                rel_entID_labels = [0] * len(entities)
                rel_entID_labels[h_entID] = 1
                rel_entID_labels[t_entID] = 1
                target["rel_entID_labels"].append(rel_entID_labels)

                head_part_labels = [0.0] * len(sent_id)
                for index in range(head_start_index, head_end_index+1):
                    head_part_labels[index] = 0.5
                head_part_labels[head_start_index] = 1.0
                head_part_labels[head_end_index] = 1.0
                target["head_part_labels"].append(head_part_labels)
                
                tail_part_labels = [0.0] * len(sent_id)
                for index in range(tail_start_index, tail_end_index+1):
                    tail_part_labels[index] = 0.5
                tail_part_labels[tail_start_index] = 1.0
                tail_part_labels[tail_end_index] = 1.0
                target["tail_part_labels"].append(tail_part_labels)

                set_head_tail.add(tuple(head[:2]))
                set_head_tail.add(tuple(tail[:2]))

            bio_labels = [0] * len(sent_id)

            entities_ = entities
            entID2entities = dict()
            if repeat_gt_entities != -1 and len(entities) > 0:
                repeat_gt_entities_ = max(repeat_gt_entities, len(entities))
                k = repeat_gt_entities_ // len(entities)
                m = repeat_gt_entities_ % len(entities)
                entities_ = entities * k
                entities_ += entities[:m]

                entID2entities = dict([(i, [len(entities)*j + i for j in range(k)])
                                    for i in range(len(entities))])
                for entID in entID2entities:
                    if entID < m:
                        entID2entities[entID].append(len(entities)*k + entID)
            # print(entID2entities)

            for ent in entities_:
                ent_type = ent[-1]

                ent_type_id = entity_type_alphabet.get_index(ent_type)
                ent_start_index, ent_end_index = ent[0], ent[1]

                max_len_mention = max(max_len_mention, ent_end_index - ent_start_index + 1)

                target["ent_type"].append(ent_type_id)
                target["ent_start_index"].append(ent_start_index)
                target["ent_end_index"].append(ent_end_index)
                target["relID_labels"].append(span2relID[tuple(ent[:2])])
                target["relID_head_labels"].append(span2relID_head[tuple(ent[:2])])
                target["relID_tail_labels"].append(span2relID_tail[tuple(ent[:2])])

                ent_have_rel = 1 if ent[:2] in set_head_tail else 0
                target["ent_have_rel"].append(ent_have_rel)

                bio_labels[ent_start_index] = 1
                for index in range(ent_start_index+1, ent_end_index+1):
                    bio_labels[index] = 2

                ent_part_labels = [0.0] * len(sent_id)
                for index in range(ent_start_index, ent_end_index+1):
                    ent_part_labels[index] = 0.5
                ent_part_labels[ent_start_index] = 1.0
                ent_part_labels[ent_end_index] = 1.0
                target["ent_part_labels"].append(ent_part_labels)

            for _ in range(repeat_num):
                samples.append([idx, sent_id, target, bio_labels, text, bep_to_char, sent_seg_encoding, context2token_masks, token_masks, entID2entities, relID2triples])

            total_triples += len(triples)
            max_triples = max(max_triples, len(triples))
            total_entities += len(entities)
            max_entities = max(max_entities, len(entities))
            num_samples += 1

    print('[num samples]:', num_samples)
    print('[avg triples]:', total_triples / num_samples)
    print('[max triples]:', max_triples)
    print('[avg entities]:', total_entities / num_samples)
    print('[max entities]:', max_entities)
    print('[max len mention]:', max_len_mention)
    print()

    return samples

def ACE_data_process(input_doc, relational_alphabet, entity_type_alphabet, tokenizer, evaluate, repeat_gt_entities=-1, repeat_gt_triples=-1,
                        max_num_subwords=200, sym_relations=[], reverse=False):

    samples = []
    total_triples = 0
    max_triples = 0
    total_entities = 0
    max_entities = 0
    max_len_mention = 0
    num_samples = 0
    print(input_doc + (' [Reverse]' if reverse else ''))

    f = open(input_doc, "r", encoding='utf-8')
    for idx, line in enumerate(f):
        data = json.loads(line)

        sentences = data['sentences']
        ners = data['ner']
        relations = data['relations']

        sentence_boundaries = [0]
        words = []
        L = 0
        for i in range(len(sentences)):
            L += len(sentences[i])
            sentence_boundaries.append(L)
            words += sentences[i]

        tokens = [tokenizer.tokenize(w) for w in words]
        subwords = [w for li in tokens for w in li]
        token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
        subword_to_word = dict()
        cum_len = 0
        for i, list_subword in enumerate(tokens):
            for j in range(cum_len, cum_len+len(list_subword)):
                subword_to_word[j] = i
            cum_len += len(list_subword)
        
        subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]

        # print(words, '\n')
        # print(tokens, '\n')
        # print(subwords, '\n')
        # print(token2subword, '\n')
        # print(subword_sentence_boundaries, '\n')

        assert len(subword_sentence_boundaries) == len(sentences) + 1
        for n in range(len(subword_sentence_boundaries) - 1):
            sentence = sentences[n]
            entities = ners[n]
            triples = relations[n]

            doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]
            sent_tokens = subwords[doc_sent_start: doc_sent_end]

            left_context_length = subword_sentence_boundaries[n] - subword_sentence_boundaries[n-1] if n > 0 else 0
            right_context_length = subword_sentence_boundaries[n+2] - subword_sentence_boundaries[n+1] if n < len(subword_sentence_boundaries)-2 else 0

            # left_length = doc_sent_start
            # right_length = len(subwords) - doc_sent_end
            sentence_length = doc_sent_end - doc_sent_start

            half_context_length = int((max_num_subwords - sentence_length) / 2)

            # assert sentence_length <= max_num_subwords
            # if sentence_length <= max_num_subwords:
            #     if left_length < right_length:
            #         left_context_length = min(left_length, half_context_length)
            #         right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
            #     else:
            #         right_context_length = min(right_length, half_context_length)
            #         left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)

            doc_offset = doc_sent_start - left_context_length
            target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
            target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]

            # print('[sentence]', sentence)
            # print('[target_tokens]')
            # print(target_tokens)

            subword_to_word_ = dict()
            for k, v in subword_to_word.items():
                subword_to_word_[k - doc_sent_start] = v - subword_to_word[doc_sent_start]

            sent_id = tokenizer.convert_tokens_to_ids(target_tokens)
            sent_seg_encoding = [0] * (1 + left_context_length) + [0] * sentence_length + [0] * (right_context_length + 1)
            assert len(sent_seg_encoding) == len(sent_id)
            word_spans = [(pos, pos+1) for pos in range(1 + left_context_length, 1 + left_context_length + sentence_length)]
            context2token_masks = torch.stack([create_word_mask(word_span[0], word_span[1], len(sent_id)) for word_span in word_spans])
            token_masks = [1] * sentence_length

            target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": [],
                    "head_mention": [], "tail_mention": [], 'head_part_labels': [], 'tail_part_labels': [], 'head_type': [], 'tail_type': [],
                    "head_entID": [], "tail_entID": [], "rel_entID_labels": [],
                    "ent_type": [], "ent_start_index": [], "ent_end_index": [], "ent_part_labels": [], 'ent_have_rel': [],
                    "relID_labels": [], "relID_head_labels": [], "relID_tail_labels": []
                }

            if evaluate:
                inverse_triples = []
                for triple in triples:
                    if triple[-1] in sym_relations:
                        inverse_triple = [triple[2], triple[3], triple[0], triple[1], triple[-1]]
                        assert inverse_triple not in triples
                        inverse_triples.append(inverse_triple)
                triples += inverse_triples

                for triple in triples:
                    head_s = token2subword[triple[0]] - doc_sent_start
                    head_e = token2subword[triple[1]+1] - doc_sent_start - 1
                    tail_s = token2subword[triple[2]] - doc_sent_start
                    tail_e = token2subword[triple[3]+1] - doc_sent_start - 1
                    head_s_, head_e_ = subword_to_word_[head_s], subword_to_word_[head_e]
                    tail_s_, tail_e_ = subword_to_word_[tail_s], subword_to_word_[tail_e]
                    h_mention = ' '.join(sentence[head_s_: head_e_+1])
                    t_mention = ' '.join(sentence[tail_s_: tail_e_+1])
                    relation_id = relational_alphabet.get_index(triple[-1])

                    target["relation"].append(relation_id)
                    target["head_start_index"].append(head_s_)
                    target["head_end_index"].append(head_e_)
                    target["tail_start_index"].append(tail_s_)
                    target["tail_end_index"].append(tail_e_)
                    target["head_mention"].append(h_mention)
                    target["tail_mention"].append(t_mention)
                    # print((h_mention, t_mention), triple[-1])

                for ent in entities:
                    sub_s = token2subword[ent[0]] - doc_sent_start
                    sub_e = token2subword[ent[1]+1] - doc_sent_start - 1
                    ent_s, ent_e = subword_to_word_[sub_s], subword_to_word_[sub_e]
                    ent_type = ent[2]
                    sub_label = entity_type_alphabet.get_index(ent_type)

                    max_len_mention = max(max_len_mention, sub_e - sub_s + 1)

                    target["ent_type"].append(sub_label)
                    target["ent_start_index"].append(ent_s)
                    target["ent_end_index"].append(ent_e)
                #     print(sentence[subword_to_word_[sub_s]: subword_to_word_[sub_e]+1], sent_tokens[sub_s: sub_e+1], ent_type)
                # print()

                samples.append([len(samples), sent_id, target, None, sentence, subword_to_word_, sent_seg_encoding, context2token_masks, token_masks, None, None])

            else:
                span2etype = {}
                span2entID = {}
                for entID, ent in enumerate(entities):
                    span2etype[tuple(ent[:2])] = entity_type_alphabet.get_index(ent[2])
                    span2entID[tuple(ent[:2])] = entID

                inverse_triples = []
                for triple in triples:
                    if triple[-1] in sym_relations:
                        inverse_triple = [triple[2], triple[3], triple[0], triple[1], triple[-1]]
                        # else:
                        #     inverse_triple = [triple[2], triple[3], triple[0], triple[1], '[Inverse]_' + triple[-1]]
                        assert inverse_triple not in triples
                        inverse_triples.append(inverse_triple)
                triples += inverse_triples

                span2relID_head = {}
                span2relID_tail = {}
                span2relID = {}
                for span in span2entID.keys():
                    span2relID_head[span] = [0] * len(triples)
                    span2relID_tail[span] = [0] * len(triples)
                    span2relID[span] = [0] * len(triples)
                    for relID, triple in enumerate(triples):
                        if span == (triple[0], triple[1]):
                            span2relID_head[span][relID] = 1
                            span2relID[span][relID] = 1
                        if span == (triple[2], triple[3]):
                            span2relID_tail[span][relID] = 1
                            span2relID[span][relID] = 1

                repeat_num = 1
                set_head_tail = set()

                # print(entities)
                # print(triples)

                triples_ = triples
                relID2triples = dict()
                if repeat_gt_triples != -1 and len(triples) > 0:
                    repeat_gt_triples_ = max(repeat_gt_triples, len(triples))
                    k = repeat_gt_triples_ // len(triples)
                    m = repeat_gt_triples_ % len(triples)
                    triples_ = triples * k

                    relID2triples = dict([(i, [len(triples)*j + i for j in range(k)])
                                        for i in range(len(triples))])

                for triple in triples_:
                    head_s = token2subword[triple[0]] - doc_sent_start
                    head_e = token2subword[triple[1]+1] - doc_sent_start - 1
                    tail_s = token2subword[triple[2]] - doc_sent_start
                    tail_e = token2subword[triple[3]+1] - doc_sent_start - 1
                    h_mention = ' '.join(sentence[subword_to_word_[head_s]: subword_to_word_[head_e]+1])
                    t_mention = ' '.join(sentence[subword_to_word_[tail_s]: subword_to_word_[tail_e]+1])
                    h_etype = span2etype[tuple(triple[0:2])]
                    t_etype = span2etype[tuple(triple[2:4])]
                    h_entID = span2entID[tuple(triple[0:2])]
                    t_entID = span2entID[tuple(triple[2:4])]

                    relation_id = relational_alphabet.get_index(triple[-1])

                    target["relation"].append(relation_id)
                    target["head_start_index"].append(head_s)
                    target["head_end_index"].append(head_e)
                    target["tail_start_index"].append(tail_s)
                    target["tail_end_index"].append(tail_e)
                    target["head_mention"].append(h_mention)
                    target["tail_mention"].append(t_mention)
                    target["head_type"].append(h_etype)
                    target["tail_type"].append(t_etype)
                    target["head_entID"].append(h_entID)
                    target["tail_entID"].append(t_entID)

                    rel_entID_labels = [0] * len(entities)
                    rel_entID_labels[h_entID] = 1
                    rel_entID_labels[t_entID] = 1
                    target["rel_entID_labels"].append(rel_entID_labels)

                    head_part_labels = [0.0] * sentence_length
                    for index in range(head_s, head_e+1):
                        head_part_labels[index] = 0.5
                    head_part_labels[head_s] = 1.0
                    head_part_labels[head_e] = 1.0
                    target["head_part_labels"].append(head_part_labels)
                    
                    tail_part_labels = [0.0] * sentence_length
                    for index in range(tail_s, tail_e+1):
                        tail_part_labels[index] = 0.5
                    tail_part_labels[tail_s] = 1.0
                    tail_part_labels[tail_e] = 1.0
                    target["tail_part_labels"].append(tail_part_labels)

                    set_head_tail.add((head_s, head_e))
                    set_head_tail.add((tail_s, tail_e))

                bio_labels = [0] * sentence_length

                entities_ = entities
                entID2entities = dict()
                if repeat_gt_entities != -1 and len(entities) > 0:
                    repeat_gt_entities_ = max(repeat_gt_entities, len(entities))
                    k = repeat_gt_entities_ // len(entities)
                    m = repeat_gt_entities_ % len(entities)
                    entities_ = entities * k
                    entities_ += entities[:m]

                    entID2entities = dict([(i, [len(entities)*j + i for j in range(k)])
                                        for i in range(len(entities))])
                    for entID in entID2entities:
                        if entID < m:
                            entID2entities[entID].append(len(entities)*k + entID)
                # print(entID2entities)

                for ent in entities_:
                    sub_s = token2subword[ent[0]] - doc_sent_start
                    sub_e = token2subword[ent[1]+1] - doc_sent_start - 1
                    ent_type = ent[2]
                    sub_label = entity_type_alphabet.get_index(ent_type)

                    max_len_mention = max(max_len_mention, sub_e - sub_s + 1)

                    target["ent_type"].append(sub_label)
                    target["ent_start_index"].append(sub_s)
                    target["ent_end_index"].append(sub_e)
                    target["relID_labels"].append(span2relID[tuple(ent[:2])])
                    target["relID_head_labels"].append(span2relID_head[tuple(ent[:2])])
                    target["relID_tail_labels"].append(span2relID_tail[tuple(ent[:2])])
                    # print(span2relID[tuple(ent[:2])])

                    ent_have_rel = 1 if (sub_s, sub_e) in set_head_tail else 0
                    target["ent_have_rel"].append(ent_have_rel)

                    bio_labels[sub_s] = 1
                    for index in range(sub_s+1, sub_e+1):
                        bio_labels[index] = 2

                    ent_part_labels = [0.0] * sentence_length
                    for index in range(sub_s, sub_e+1):
                        ent_part_labels[index] = 0.5
                    ent_part_labels[sub_s] = 1.0
                    ent_part_labels[sub_e] = 1.0
                    target["ent_part_labels"].append(ent_part_labels)

                for _ in range(repeat_num):
                    if not reverse or len(triples) > 0:
                        samples.append([len(samples), sent_id, target, bio_labels, sentence, subword_to_word_, sent_seg_encoding, context2token_masks, token_masks, entID2entities, relID2triples])

                # print('len', len(target['relation']))
                # print(target['relation'])
                # print('', flush=True)

            if not reverse or len(triples) > 0:
                total_triples += len(triples)
                max_triples = max(max_triples, len(triples))
                total_entities += len(entities)
                max_entities = max(max_entities, len(entities))
                num_samples += 1

    print('[num samples]:', num_samples)
    print('[avg triples]:', total_triples / num_samples)
    print('[max triples]:', max_triples)
    print('[avg entities]:', total_entities / num_samples)
    print('[max entities]:', max_entities)
    print('[max len mention]:', max_len_mention)
    print()

    return samples


class Data:
    def __init__(self):
        self.relational_alphabet = Alphabet("Relation", unkflag=False, padflag=False)
        self.train_loader = []
        self.valid_loader = []
        self.test_loader = []
        self.weight = {}

        self.entity_type_alphabet = Alphabet("Entity", unkflag=False, padflag=False)

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Relation Alphabet Size: %s" % self.relational_alphabet.size())
        print("     Ent Type Alphabet Size: %s" % self.entity_type_alphabet.size())
        print("     Train  Instance Number: %s" % (len(self.train_loader)))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args):
        if args.dataset_name == 'child':
            tokenizer = BertTokenizerFast.from_pretrained(args.bert_directory, do_lower_case=True)
            self.train_loader = child_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=False,
                                    repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
            self.valid_loader = child_data_process(args.valid_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
            self.test_loader = child_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)

        elif args.dataset_name == 'Text2DT':
            tokenizer = BertTokenizerFast.from_pretrained(args.bert_directory, do_lower_case=False)
            self.train_loader = text2dt_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=False,
                                    repeat_gt_entities=25, repeat_gt_triples=15)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
            self.valid_loader = text2dt_data_process(args.valid_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
            self.test_loader = text2dt_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
        elif args.dataset_name == 'biorelex':
            tokenizer = BertTokenizerFast.from_pretrained(args.bert_directory, do_lower_case=True)
            self.train_loader = biorelex_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=False,
                                    repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
            self.valid_loader = biorelex_data_process(args.valid_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
            self.test_loader = biorelex_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
        elif args.dataset_name == 'ade':
            tokenizer = BertTokenizerFast.from_pretrained(args.bert_directory, do_lower_case=True)
            self.train_loader = ade_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=False,
                                    repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples)
            self.weight = copy.deepcopy(self.relational_alphabet.index_num)
            self.valid_loader = ade_data_process(args.valid_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
            self.test_loader = ade_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True)
        else:
            sym_relations = []
            if args.dataset_name == 'ace2005':
                sym_relations = ['PER-SOC']
            if args.dataset_name == 'SciERC':
                sym_relations = ['CONJUNCTION', 'COMPARE']

            tokenizer = BertTokenizer.from_pretrained(args.bert_directory, do_lower_case=True)
            
            if "train_file" in args:
                self.train_loader = ACE_data_process(args.train_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=False, repeat_gt_entities=args.repeat_gt_entities, repeat_gt_triples=args.repeat_gt_triples, sym_relations=sym_relations)
                self.weight = copy.deepcopy(self.relational_alphabet.index_num)
            if "valid_file" in args:
                self.valid_loader = ACE_data_process(args.valid_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True, sym_relations=sym_relations)
            if "test_file" in args:
                self.test_loader = ACE_data_process(args.test_file, self.relational_alphabet, self.entity_type_alphabet, tokenizer, evaluate=True, sym_relations=sym_relations)

        self.relational_alphabet.close()
        self.entity_type_alphabet.close()


def build_data(args):

    file = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    # if os.path.exists(file) and not args.refresh:
    #     data = load_data_setting(args)
    # else:
    data = Data()
    data.generate_instance(args)
    save_data_setting(data, args)
    return data


def save_data_setting(data, args):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.generated_data_directory):
        os.makedirs(args.generated_data_directory)
    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", saved_path)


def load_data_setting(args):

    saved_path = args.generated_data_directory + args.dataset_name + "_" + args.model_name + "_data.pickle"
    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)
    data.show_data_summary()
    return data


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor

def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def create_word_mask(start, end, context_size):
        mask = torch.zeros(context_size, dtype=torch.bool)
        mask[start:end] = 1
        return mask
