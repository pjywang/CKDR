import re
import os
from collections import defaultdict

def parse_taxonomic_string(taxa_string):
    """
    Parse a taxonomic string to extract genus and species information.
    Returns (genus, species) tuple with proper bracket handling.
    """
    levels = taxa_string.split(';')
    
    genus = None
    species = None
    
    for level in levels:
        if level.startswith('g__'):
            genus_name = level[3:]  # Remove 'g__' prefix
            if genus_name and genus_name != '':
                # Handle bracketed genus names
                bracket_match = re.search(r'\[([^\]]+)\]', genus_name)
                if bracket_match:
                    genus = bracket_match.group(1)
                else:
                    genus = genus_name
        elif level.startswith('s__'):
            species_name = level[3:]  # Remove 's__' prefix
            if species_name and species_name != '':
                # Handle bracketed species names - extract the actual species
                bracket_match = re.search(r'\[([^\]]+)\]', species_name)
                if bracket_match:
                    # For bracketed species, use the bracketed name as genus
                    genus = bracket_match.group(1)
                    species = None  # No specific species info
                else:
                    species = species_name.replace('_', ' ')
    
    return genus, species

def extract_genus_from_taxonomic_string(taxa_string):
    """
    Extract genus name from any part of the taxonomic string, handling all bracket cases.
    """
    levels = taxa_string.split(';')
    
    # First try to get genus from g__ level
    for level in levels:
        if level.startswith('g__'):
            genus_name = level[3:]
            if genus_name and genus_name != '':
                # Handle bracketed genus names
                bracket_match = re.search(r'\[([^\]]+)\]', genus_name)
                if bracket_match:
                    return bracket_match.group(1)
                else:
                    return genus_name
    
    # If no g__ level, check species level for bracketed names
    for level in levels:
        if level.startswith('s__'):
            species_name = level[3:]
            if species_name and species_name != '':
                # Check for bracketed names in species level
                bracket_match = re.search(r'\[([^\]]+)\]', species_name)
                if bracket_match:
                    return bracket_match.group(1)
    
    # If still no genus found, try to extract from any meaningful level
    for level in reversed(levels):
        if '__' in level and not level.endswith('__'):
            level_name = level.split('__')[1]
            if level_name:
                # Clean up and check for brackets
                bracket_match = re.search(r'\[([^\]]+)\]', level_name)
                if bracket_match:
                    return bracket_match.group(1)
                else:
                    # Return the first word if it looks like a genus
                    clean_name = level_name.replace('_', ' ')
                    first_word = clean_name.split()[0]
                    if first_word and first_word[0].isupper():
                        return first_word
    
    return None

def group_taxa_by_genus(taxa_list, top=15):
    """
    Group taxa by genus with consistent bracket handling.
    """
    genus_counts = defaultdict(int)
    unclassified_taxa = []
    
    for taxa_string in taxa_list:
        genus = extract_genus_from_taxonomic_string(taxa_string)
        
        if genus:
            genus_counts[genus] += 1
        else:
            # Handle completely unclassified taxa
            levels = taxa_string.split(';')
            # Find the most specific level
            meaningful_level = None
            for level in reversed(levels):
                if '__' in level and not level.endswith('__'):
                    meaningful_level = level.split('__')[1]
                    break
            
            if meaningful_level:
                clean_name = meaningful_level.replace('_', ' ')
                # Remove any remaining brackets for display
                clean_name = re.sub(r'\[([^\]]+)\]', r'\1', clean_name)
                unclassified_taxa.append(clean_name)
            else:
                unclassified_taxa.append("Unknown taxon")
    
    # Format the results
    formatted_taxa = []
    
    # Process genus groups (sorted by count in decreasing order, then alphabetically)
    sorted_genera = sorted(genus_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Show only top (15) most frequent genera
    top_genera = sorted_genera[:top]
    remaining_genera_count = len(sorted_genera) - top
    
    for genus, count in top_genera:
        if count == 1:
            formatted_taxa.append(f"\\textit{{{genus}}}")
        else:
            formatted_taxa.append(f"\\textit{{{genus}}} ({count})")
    
    # Add indication of remaining genera if there are more than top
    if remaining_genera_count > 0:
        formatted_taxa.append(f"and {remaining_genera_count} other genera")
    
    # Add unclassified taxa (but don't count them towards the top limit)
    for taxon in sorted(set(unclassified_taxa)):
        # Format family/order level names appropriately
        if any(suffix in taxon.lower() for suffix in ['aceae', 'ales', 'ia']):
            formatted_taxa.append(f"\\textit{{{taxon}}}")
        else:
            formatted_taxa.append(taxon)
    
    return formatted_taxa

def generate_taxa_table(cluster_dict, output_path, dataname, top=15):
    """
    Generate a LaTeX booktabs table showing taxonomic composition of clusters.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    latex_content = []
    
    # Table header
    latex_content.append("\\begin{table}[ht]")
    latex_content.append("\\centering")
    latex_content.append(f"\\caption{{{dataname} microbiome data}}")
    latex_content.append(f"\\label{{tab:{dataname.lower()}_taxa}}")
    latex_content.append("\\begin{tabular}{p{1.2cm}p{12cm}}")
    latex_content.append("\\toprule")
    latex_content.append("Cluster & Taxa \\\\")
    latex_content.append("\\midrule")
    
    # Process each cluster
    for cluster_id in sorted(cluster_dict.keys()):
        taxa_list = cluster_dict[cluster_id]
        
        if len(taxa_list) == 0:
            latex_content.append(f"$z_{{{cluster_id}}}$ & (No taxa assigned) \\\\")
            continue
        
        # Group and format taxa
        formatted_taxa = group_taxa_by_genus(taxa_list, top=top)
        
        # Create a single string for this cluster
        taxa_text = "; ".join(formatted_taxa)
        
        # Add total count
        taxa_text += f" (\\textbf{{{len(taxa_list)} total}})"
        
        # Add the row
        latex_content.append(f"$z_{{{cluster_id}}}$ & {taxa_text} \\\\")
        
        # Add spacing between clusters (except for the last one)
        if cluster_id != max(cluster_dict.keys()):
            latex_content.append("\\addlinespace")
    
    # Table footer
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX table saved to: {output_path}")
    return latex_content