import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_columns(df):
    """
    Parses the dataframe columns to identify ranking columns and extract metadata.
    Returns a dictionary mapping column names to (Category, Course).
    """
    col_map = {}
    # Regex to capture Category and Course Name
    # matches: "... - Ranks - Category - Course - Rank"
    # We use non-greedy matching for the prefix if needed, but the anchor is " - Ranks - "
    pattern = re.compile(r".* - Ranks - (Most Beneficial|Neutral|Least Beneficial) - (.*) - Rank")

    for col in df.columns:
        match = pattern.search(col)
        if match:
            category = match.group(1)
            course = match.group(2).strip()
            col_map[col] = (category, course)
    return col_map

def calculate_rankings(df, col_map):
    """
    Calculates the global rank for each course for each student.
    Returns a DataFrame with columns ['Student', 'Course', 'Global_Rank']
    """
    records = []

    # Iterate over each student (row)
    for idx, row in df.iterrows():
        student_id = row.get('Response ID', idx)

        # Buckets for this student
        most_beneficial = []
        neutral = []
        least_beneficial = []

        # Go through each column mapped to a ranking
        for col, (category, course) in col_map.items():
            rank_val = row[col]
            if pd.notna(rank_val):
                if category == 'Most Beneficial':
                    most_beneficial.append((course, rank_val))
                elif category == 'Neutral':
                    neutral.append((course, rank_val))
                elif category == 'Least Beneficial':
                    least_beneficial.append((course, rank_val))

        # Sort each bucket by the inner rank (1, 2, 3...)
        most_beneficial.sort(key=lambda x: x[1])
        neutral.sort(key=lambda x: x[1])
        least_beneficial.sort(key=lambda x: x[1])

        # Assign Global Ranks
        current_rank = 1

        # Add Most Beneficial
        for course, _ in most_beneficial:
            records.append({
                'Student': student_id,
                'Course': course,
                'Global_Rank': current_rank
            })
            current_rank += 1

        # Add Neutral
        for course, _ in neutral:
            records.append({
                'Student': student_id,
                'Course': course,
                'Global_Rank': current_rank
            })
            current_rank += 1

        # Add Least Beneficial
        for course, _ in least_beneficial:
            records.append({
                'Student': student_id,
                'Course': course,
                'Global_Rank': current_rank
            })
            current_rank += 1

    return pd.DataFrame(records)

def main():
    # Paths
    data_path = 'data/data.xlsx'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Load Data
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return

    # Parse Columns
    col_map = parse_columns(df)
    if not col_map:
        print("No ranking columns found.")
        return

    print(f"Found {len(col_map)} ranking columns.")

    # Calculate Rankings
    print("Calculating global rankings...")
    rankings_df = calculate_rankings(df, col_map)

    if rankings_df.empty:
        print("No rankings could be calculated.")
        return

    # Aggregate Rankings
    # Calculate Mean Global Rank for each course
    course_stats = rankings_df.groupby('Course')['Global_Rank'].agg(['mean', 'count', 'std']).reset_index()
    course_stats = course_stats.sort_values(by='mean')

    # Rename for clarity
    course_stats = course_stats.rename(columns={'mean': 'Average Rank', 'count': 'N', 'std': 'Std Dev'})

    # Save Ranking Text
    ranking_text_path = os.path.join(output_dir, 'ranking.txt')
    with open(ranking_text_path, 'w') as f:
        f.write("Course Ranking (Based on Average Global Rank - Lower is Better)\n")
        f.write("================================================================\n\n")
        # Format the float columns
        f.write(course_stats.to_string(index=False, float_format="%.2f"))

    print(f"Ranking text saved to {ranking_text_path}")
    print(course_stats)

    # Visualization
    print("Generating visualization...")
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Create barplot
    # We want to display the courses sorted by average rank.
    # The 'course_stats' df is already sorted by 'Average Rank' ascending.
    # sns.barplot respects the order of the data if x/y are categorical, or we can enforce order.

    # Color palette
    palette = sns.color_palette("viridis", len(course_stats))

    ax = sns.barplot(
        x='Average Rank',
        y='Course',
        data=course_stats,
        palette=palette,
        hue='Course',
        order=course_stats['Course'] # Explicitly set order
    )

    plt.title('Average Course Ranking (Lower Rank = More Beneficial)', fontsize=16)
    plt.xlabel('Average Global Rank', fontsize=12)
    plt.ylabel('Course', fontsize=12)

    # Add value labels to the bars
    for i, v in enumerate(course_stats['Average Rank']):
        ax.text(v + 0.1, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rank_order.png')
    plt.savefig(plot_path)
    print(f"Ranking plot saved to {plot_path}")

if __name__ == "__main__":
    main()
