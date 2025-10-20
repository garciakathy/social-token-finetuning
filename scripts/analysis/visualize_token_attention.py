#!/usr/bin/env python3
"""
Visualize token-to-token attention patterns for social tokens.
Shows which tokens attend to SOC_G and SOC_L tokens.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_special_token_positions(input_ids, tokenizer, soc_g_id, soc_l_id):
    """Find positions of SOC_G and SOC_L tokens in sequence."""
    positions = {
        'soc_g': [],
        'soc_l': [],
        'text': []
    }

    for i, token_id in enumerate(input_ids):
        if token_id == soc_g_id:
            positions['soc_g'].append(i)
        elif token_id == soc_l_id:
            positions['soc_l'].append(i)
        elif token_id != tokenizer.pad_token_id:
            positions['text'].append(i)

    return positions


def extract_attention_weights(model, input_ids, attention_mask, inputs_embeds):
    """Extract attention weights from all layers."""
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

    # outputs.attentions is tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
    return outputs.attentions


def plot_attention_heatmap(attention, input_ids, tokenizer, positions, layer_idx, head_idx, save_path):
    """Plot attention heatmap for a single layer and head."""

    # Average over batch dimension if needed
    if attention.dim() == 4:
        attn = attention[0, head_idx].cpu().numpy()  # [seq_len, seq_len]
    else:
        attn = attention[head_idx].cpu().numpy()

    seq_len = attn.shape[0]

    # Create token labels
    tokens = []
    token_types = []
    for i, token_id in enumerate(input_ids[:seq_len]):
        token = tokenizer.decode([token_id])
        if i in positions['soc_g']:
            tokens.append('[SOC-G]')
            token_types.append('global')
        elif i in positions['soc_l']:
            tokens.append('[SOC-L]')
            token_types.append('local')
        else:
            tokens.append(token[:15])  # Truncate long tokens
            token_types.append('text')

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot heatmap
    sns.heatmap(attn, cmap='viridis', square=True,
                xticklabels=tokens, yticklabels=tokens,
                cbar_kws={'label': 'Attention Weight'},
                vmin=0, vmax=attn.max())

    # Color code x and y labels
    for i, (xtick, ytick) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        if token_types[i] == 'global':
            xtick.set_color('red')
            ytick.set_color('red')
            xtick.set_weight('bold')
            ytick.set_weight('bold')
        elif token_types[i] == 'local':
            xtick.set_color('orange')
            ytick.set_color('orange')
            xtick.set_weight('bold')
            ytick.set_weight('bold')

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlabel('Key (attended to)', fontsize=12, fontweight='bold')
    plt.ylabel('Query (attending from)', fontsize=12, fontweight='bold')
    plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}\n' +
              f'Red=[SOC-G], Orange=[SOC-L], Black=Text',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def compute_attention_statistics(attentions, positions, tokenizer):
    """Compute statistics about attention to social tokens."""

    stats = {
        'layer': [],
        'avg_attn_to_soc_g': [],
        'avg_attn_to_soc_l': [],
        'avg_attn_to_text': [],
        'max_attn_to_soc_g': [],
        'max_attn_to_soc_l': []
    }

    for layer_idx, layer_attn in enumerate(attentions):
        # Average over heads and batch
        if layer_attn.dim() == 4:
            attn = layer_attn[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
        else:
            attn = layer_attn.mean(dim=0).cpu().numpy()

        # Compute average attention TO social tokens (across all query positions)
        if positions['soc_g']:
            avg_to_g = attn[:, positions['soc_g']].mean()
            max_to_g = attn[:, positions['soc_g']].max()
        else:
            avg_to_g = 0
            max_to_g = 0

        if positions['soc_l']:
            avg_to_l = attn[:, positions['soc_l']].mean()
            max_to_l = attn[:, positions['soc_l']].max()
        else:
            avg_to_l = 0
            max_to_l = 0

        if positions['text']:
            avg_to_text = attn[:, positions['text']].mean()
        else:
            avg_to_text = 0

        stats['layer'].append(layer_idx)
        stats['avg_attn_to_soc_g'].append(avg_to_g)
        stats['avg_attn_to_soc_l'].append(avg_to_l)
        stats['avg_attn_to_text'].append(avg_to_text)
        stats['max_attn_to_soc_g'].append(max_to_g)
        stats['max_attn_to_soc_l'].append(max_to_l)

    return stats


def plot_attention_statistics(stats, save_path):
    """Plot attention statistics across layers."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Average attention
    ax = axes[0]
    ax.plot(stats['layer'], stats['avg_attn_to_soc_g'], 'r-o', label='SOC-G (Global)', linewidth=2)
    ax.plot(stats['layer'], stats['avg_attn_to_soc_l'], 'orange', marker='s', label='SOC-L (Local)', linewidth=2)
    ax.plot(stats['layer'], stats['avg_attn_to_text'], 'k--', label='Text tokens', linewidth=2)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Average Attention to Token Types Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Max attention
    ax = axes[1]
    ax.plot(stats['layer'], stats['max_attn_to_soc_g'], 'r-o', label='SOC-G (Global)', linewidth=2)
    ax.plot(stats['layer'], stats['max_attn_to_soc_l'], 'orange', marker='s', label='SOC-L (Local)', linewidth=2)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Maximum Attention to Social Tokens Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-dir', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--model-dir', type=str, required=True, help='Path to model')
    parser.add_argument('--projector-ckpt', type=str, required=True, help='Path to projector checkpoint')
    parser.add_argument('--captions-csv', type=str, default='data/captions.csv', help='Path to captions CSV')
    parser.add_argument('--features-dir', type=str, default='data/seamless_v4_ooo_features', help='Path to features')
    parser.add_argument('--num-examples', type=int, default=5, help='Number of examples to visualize')
    parser.add_argument('--output-dir', type=str, default='data/results/attention_viz', help='Output directory')
    parser.add_argument('--inject', type=str, default='global', choices=['global', 'full'], help='Injection mode')
    parser.add_argument('--layer', type=int, default=6, help='Layer to visualize (default: 6)')
    parser.add_argument('--head', type=int, default=0, help='Attention head to visualize (default: 0)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("TOKEN ATTENTION VISUALIZATION")
    print("=" * 80)
    print(f"Injection mode: {args.inject}")
    print(f"Layer: {args.layer}, Head: {args.head}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Get special token IDs
    soc_g_id = tokenizer.convert_tokens_to_ids('[SOC_G]')
    soc_l_id = tokenizer.convert_tokens_to_ids('[SOC_L]')

    print(f"SOC_G token ID: {soc_g_id}")
    print(f"SOC_L token ID: {soc_l_id}")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='eager'
    )
    model.eval()

    # Load projector
    print(f"\nLoading projector from {args.projector_ckpt}...")
    from scripts.llm_finetuning.next_utt_social_ooo_fix import VisualProjector

    ckpt = torch.load(args.projector_ckpt, map_location='cpu')
    if 'projector_state_dict' in ckpt:
        proj_state = ckpt['projector_state_dict']
    elif 'model_state_dict' in ckpt:
        proj_state = {k.replace('projector.', ''): v for k, v in ckpt['model_state_dict'].items() if 'projector' in k}
    else:
        proj_state = ckpt

    projector = VisualProjector(input_dim=1024, output_dim=model.config.hidden_size)
    projector.load_state_dict(proj_state)
    projector = projector.to(model.device).to(torch.bfloat16)
    projector.eval()

    # Load caption data
    print("\nLoading captions...")
    import pandas as pd
    df = pd.read_csv(args.captions_csv)

    # Process examples
    print(f"\nProcessing {args.num_examples} examples...")

    all_stats = []

    for idx in range(min(args.num_examples, len(df))):
        row = df.iloc[idx]
        uid = row['uid']
        caption = row['caption']

        print(f"\n[{idx+1}/{args.num_examples}] {uid}: {caption[:80]}...")

        # Load visual features
        feature_path = os.path.join(args.features_dir, f"{uid}.npz")
        if not os.path.exists(feature_path):
            print(f"  Skipping (features not found)")
            continue

        features = np.load(feature_path)
        global_feat = torch.tensor(features['global'], dtype=torch.bfloat16).unsqueeze(0)

        # Tokenize
        enc = tokenizer(caption, return_tensors='pt', padding=False, truncation=True, max_length=512)
        input_ids = enc['input_ids'].to(model.device)
        attention_mask = enc['attention_mask'].to(model.device)

        # Get text embeddings
        text_embeds = model.model.embed_tokens(input_ids)

        # Project visual features
        global_proj = projector(global_feat.to(model.device))

        # Inject social tokens
        if args.inject == 'global':
            # Prepend SOC_G
            inputs_embeds = torch.cat([global_proj, text_embeds], dim=1)
            # Update attention mask
            attention_mask = torch.cat([
                torch.ones((1, 1), dtype=attention_mask.dtype, device=model.device),
                attention_mask
            ], dim=1)
            # Update input_ids for visualization
            soc_g_tensor = torch.tensor([[soc_g_id]], device=model.device)
            input_ids = torch.cat([soc_g_tensor, input_ids], dim=1)

        elif args.inject == 'full':
            # Load local features
            if 'local' in features:
                local_feats = torch.tensor(features['local'], dtype=torch.bfloat16)
                local_proj = projector(local_feats.to(model.device))

                # Get POS tags
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(caption)

                # Find noun/verb positions
                insert_positions = []
                for token in doc:
                    if token.pos_ in ['NOUN', 'VERB', 'PROPN']:
                        # Find position in tokenized sequence
                        token_start = token.idx
                        for i, tok_id in enumerate(input_ids[0]):
                            tok_text = tokenizer.decode([tok_id])
                            if tok_text.strip() and token.text.lower().startswith(tok_text.lower()):
                                insert_positions.append(i + 1)  # Insert after token
                                break

                # Insert local tokens (in reverse to maintain positions)
                embeds_list = [text_embeds[0]]
                ids_list = [input_ids[0]]
                attn_list = [attention_mask[0]]

                for pos_idx, pos in enumerate(sorted(set(insert_positions))):
                    if pos_idx < len(local_proj):
                        # Insert at position
                        embeds_list.insert(pos, local_proj[pos_idx])
                        ids_list.insert(pos, torch.tensor([soc_l_id], device=model.device))
                        attn_list.insert(pos, torch.tensor([1], device=model.device))

                text_embeds = torch.stack(embeds_list).unsqueeze(0)
                input_ids = torch.stack(ids_list).unsqueeze(0)
                attention_mask = torch.stack(attn_list).unsqueeze(0)

            # Prepend global
            inputs_embeds = torch.cat([global_proj, text_embeds], dim=1)
            attention_mask = torch.cat([
                torch.ones((1, 1), dtype=attention_mask.dtype, device=model.device),
                attention_mask
            ], dim=1)
            soc_g_tensor = torch.tensor([[soc_g_id]], device=model.device)
            input_ids = torch.cat([soc_g_tensor, input_ids], dim=1)

        # Extract attention weights
        attentions = extract_attention_weights(model, input_ids[0], attention_mask, inputs_embeds)

        # Get token positions
        positions = get_special_token_positions(input_ids[0], tokenizer, soc_g_id, soc_l_id)

        print(f"  Sequence length: {len(input_ids[0])}")
        print(f"  SOC-G positions: {positions['soc_g']}")
        print(f"  SOC-L positions: {positions['soc_l']}")
        print(f"  Text positions: {len(positions['text'])}")

        # Plot attention heatmap for specified layer and head
        save_path = os.path.join(args.output_dir, f"{uid}_layer{args.layer}_head{args.head}.png")
        plot_attention_heatmap(
            attentions[args.layer],
            input_ids[0],
            tokenizer,
            positions,
            args.layer,
            args.head,
            save_path
        )

        # Compute statistics
        stats = compute_attention_statistics(attentions, positions, tokenizer)
        all_stats.append(stats)

        # Plot statistics
        stats_path = os.path.join(args.output_dir, f"{uid}_stats.png")
        plot_attention_statistics(stats, stats_path)

    # Average statistics across examples
    if all_stats:
        print("\nComputing average statistics across examples...")
        avg_stats = {
            'layer': all_stats[0]['layer'],
            'avg_attn_to_soc_g': np.mean([s['avg_attn_to_soc_g'] for s in all_stats], axis=0).tolist(),
            'avg_attn_to_soc_l': np.mean([s['avg_attn_to_soc_l'] for s in all_stats], axis=0).tolist(),
            'avg_attn_to_text': np.mean([s['avg_attn_to_text'] for s in all_stats], axis=0).tolist(),
            'max_attn_to_soc_g': np.mean([s['max_attn_to_soc_g'] for s in all_stats], axis=0).tolist(),
            'max_attn_to_soc_l': np.mean([s['max_attn_to_soc_l'] for s in all_stats], axis=0).tolist()
        }

        avg_path = os.path.join(args.output_dir, f"average_stats_{args.inject}.png")
        plot_attention_statistics(avg_stats, avg_path)

        # Save to CSV
        import pandas as pd
        df_stats = pd.DataFrame(avg_stats)
        csv_path = os.path.join(args.output_dir, f"average_stats_{args.inject}.csv")
        df_stats.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    print("\n" + "=" * 80)
    print("DONE! Visualizations saved to:", args.output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
