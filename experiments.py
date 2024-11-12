from typing import List, Dict, Any

def get_experiments() -> List[Dict[str, Any]]:
    """Return a list of experiment configurations."""

    base_columns = ['Winner']

    return [
        # Baseline experiment with essential features
        {
            'name': 'baseline',
            'params': {
                'columns_to_keep': base_columns + [
                    'fights_A', 'fights_B',
                    'wins_A', 'wins_B',
                    'losses_A', 'losses_B',
                    'current_win_streak_A', 'current_win_streak_B',
                    'current_loss_streak_A', 'current_loss_streak_B',
                    'longest_win_streak_A', 'longest_win_streak_B',
                    'longest_loss_streak_A', 'longest_loss_streak_B'
                ],
                'encoding_columns': {
                    'Weight_Class': ('onehot', {}),
                    'STANCE_A': ('onehot', {}),
                    'STANCE_B': ('onehot', {}),
                },
                'diff_columns': {
                    'Height': True,
                    'Reach': True,
                    'Weight': True,
                },
                'mirror': 'whole',
                'x_scale': True,
            }
        }
    ]
