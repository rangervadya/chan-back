import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../utils/app_localizations.dart';

class GuideScreen extends StatefulWidget {
  const GuideScreen({super.key});

  @override
  State<GuideScreen> createState() => _GuideScreenState();
}

class _GuideScreenState extends State<GuideScreen> {
  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  }

  @override
  Widget build(BuildContext context) {
    final l = AppLocalizations.tr;
    return Scaffold(
      backgroundColor: const Color(0xFF151412),
      appBar: AppBar(
        backgroundColor: const Color(0xFF151412),
        foregroundColor: Colors.white,
        title: Text(l(context, 'guide_title')),
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back, color: Colors.white, size: 22),
          style: IconButton.styleFrom(
            backgroundColor: Colors.white12,
            shape: const CircleBorder(),
            fixedSize: const Size(44, 44), // ваш размер
            padding: EdgeInsets.zero,
          ),
        ),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeader(),
                const SizedBox(height: 32),
                _buildSection(l(context, 'section_creating_project'), [
                  _buildBullet(l(context, 'guide_take_photo')),
                  _buildBullet(l(context, 'guide_choose_from_gallery')),
                ]),
                const SizedBox(height: 24),
                _buildSection(l(context, 'section_photo_requirements'), [
                  _buildTableRow(
                    l(context, 'photo_req_lighting'),
                    l(context, 'photo_req_lighting_desc'),
                  ),
                  _buildTableRow(
                    l(context, 'photo_req_background'),
                    l(context, 'photo_req_background_desc'),
                  ),
                  _buildTableRow(
                    l(context, 'photo_req_framing'),
                    l(context, 'photo_req_framing_desc'),
                  ),
                  _buildTableRow(
                    l(context, 'photo_req_stability'),
                    l(context, 'photo_req_stability_desc'),
                  ),
                  _buildTableRow(
                    l(context, 'photo_req_angle'),
                    l(context, 'photo_req_angle_desc'),
                  ),
                ]),
                const SizedBox(height: 24),
                _buildSection(l(context, 'section_recolor_modes'), [
                  _buildBullet(
                    '${l(context, 'mode_easy')} — ${l(context, 'mode_easy_desc')}',
                  ),
                  _buildBullet(
                    '${l(context, 'mode_complex')} — ${l(context, 'mode_complex_desc')}',
                  ),
                ]),
                const SizedBox(height: 24),
                _buildSection(l(context, 'section_common_problems'), [
                  _buildProblemRow(
                    l(context, 'prob_no_selection'),
                    l(context, 'prob_no_selection_sol'),
                  ),
                  _buildProblemRow(
                    l(context, 'prob_bg_recolored'),
                    l(context, 'prob_bg_recolored_sol'),
                  ),
                  _buildProblemRow(
                    l(context, 'prob_blurry'),
                    l(context, 'prob_blurry_sol'),
                  ),
                  _buildProblemRow(
                    l(context, 'prob_wrong_color'),
                    l(context, 'prob_wrong_color_sol'),
                  ),
                  _buildProblemRow(
                    l(context, 'prob_missing_details'),
                    l(context, 'prob_missing_details_sol'),
                  ),
                  _buildProblemRow(
                    l(context, 'prob_slow'),
                    l(context, 'prob_slow_sol'),
                  ),
                ]),
                const SizedBox(height: 24),
                _buildSection(l(context, 'section_tips'), [
                  _buildBullet(l(context, 'tip_daylight')),
                  _buildBullet(l(context, 'tip_no_reflections')),
                  _buildBullet(l(context, 'tip_closeup')),
                  _buildBullet(l(context, 'tip_simple_bg')),
                  _buildBullet(l(context, 'tip_start_easy')),
                  _buildBullet(l(context, 'tip_experiment_materials')),
                  _buildBullet(l(context, 'tip_thin_objects')),
                ]),
                const SizedBox(height: 32),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return const Text(
      'Сamelion',
      style: TextStyle(
        color: Color(0xFFFFC107),
        fontSize: 32,
        fontWeight: FontWeight.w800,
      ),
    );
  }

  Widget _buildSection(String title, List<Widget> children) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 12),
        ...children,
      ],
    );
  }

  Widget _buildBullet(String text) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Text(
            text,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 14,
              height: 1.5,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildTableRow(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 120,
          child: Text(
            label,
            style: const TextStyle(
              color: Color(0xFFFFC107),
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 13,
              height: 1.5,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildProblemRow(String problem, String solution) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          problem,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w600,
          ),
        ),
        Text(
          solution,
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 13,
            height: 1.5,
          ),
        ),
      ],
    );
  }
}
